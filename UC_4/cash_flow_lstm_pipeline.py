"""
Cash Flow Forecasting Pipeline using LSTM
==========================================

Tạo một pipeline hoàn chỉnh để dự đoán dòng tiền sử dụng mô hình LSTM
dựa trên dữ liệu của ngân hàng bao gồm: deposits, loans, withdrawals, 
bond sales, operating expenses và interbank transfers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

class CashFlowLSTMPipeline:
    """
    Pipeline hoàn chỉnh cho dự đoán dòng tiền sử dụng LSTM
    """
    
    def __init__(self, data_path):
        """
        Khởi tạo pipeline
        
        Args:
            data_path (str): Đường dẫn đến thư mục chứa dữ liệu CSV
        """
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.model = None
        self.sequence_length = 30  # Sử dụng 30 ngày dữ liệu để dự đoán
        
    def load_and_preprocess_data(self):
        """
        Tải và tiền xử lý tất cả dữ liệu từ các file CSV
        
        Returns:
            pd.DataFrame: DataFrame chứa dòng tiền tổng hợp theo ngày
        """
        print("Đang tải dữ liệu...")
        
        # Tải các file dữ liệu
        deposits = pd.read_csv(f"{self.data_path}/fact_deposits.csv")
        loans = pd.read_csv(f"{self.data_path}/fact_loans.csv")
        withdrawals = pd.read_csv(f"{self.data_path}/fact_withdrawals.csv")
        bond_sales = pd.read_csv(f"{self.data_path}/fact_bond_sales.csv")
        operating_expenses = pd.read_csv(f"{self.data_path}/fact_operating_expenses.csv")
        interbank_transfers = pd.read_csv(f"{self.data_path}/fact_interbank_transfers.csv")
        
        print(f"Đã tải dữ liệu:")
        print(f"  - Deposits: {len(deposits)} records")
        print(f"  - Loans: {len(loans)} records")
        print(f"  - Withdrawals: {len(withdrawals)} records")
        print(f"  - Bond Sales: {len(bond_sales)} records")
        print(f"  - Operating Expenses: {len(operating_expenses)} records")
        print(f"  - Interbank Transfers: {len(interbank_transfers)} records")
        
        # Xử lý dữ liệu deposits (cash inflow)
        deposits['Deposit_Date'] = pd.to_datetime(deposits['Deposit_Date'])
        deposits['date'] = deposits['Deposit_Date'].dt.date
        deposits_daily = deposits.groupby('date').agg({
            'Deposit_Amount': 'sum',
            'Interest_Outflow': 'sum'
        }).reset_index()
        deposits_daily['net_deposit_inflow'] = deposits_daily['Deposit_Amount'] - deposits_daily['Interest_Outflow']
        
        # Xử lý dữ liệu loans (cash outflow từ việc cho vay, inflow từ lãi)
        loans['Loan_Date'] = pd.to_datetime(loans['Loan_Date'])
        loans['date'] = loans['Loan_Date'].dt.date
        loans_daily = loans.groupby('date').agg({
            'Loan_Amount': 'sum',
            'Interest_Inflow': 'sum'
        }).reset_index()
        loans_daily['net_loan_outflow'] = -(loans_daily['Loan_Amount'] - loans_daily['Interest_Inflow'])
        
        # Xử lý dữ liệu withdrawals (cash outflow)
        withdrawals['Withdrawal_Date'] = pd.to_datetime(withdrawals['Withdrawal_Date'])
        withdrawals['date'] = withdrawals['Withdrawal_Date'].dt.date
        withdrawals_daily = withdrawals.groupby('date').agg({
            'Withdrawal_Amount': 'sum'
        }).reset_index()
        withdrawals_daily['withdrawal_outflow'] = -withdrawals_daily['Withdrawal_Amount']
        
        # Xử lý dữ liệu bond sales (cash inflow)
        bond_sales['Bond_Sale_Date'] = pd.to_datetime(bond_sales['Bond_Sale_Date'])
        bond_sales['date'] = bond_sales['Bond_Sale_Date'].dt.date
        bond_sales_daily = bond_sales.groupby('date').agg({
            'Sale_Amount': 'sum'
        }).reset_index()
        bond_sales_daily['bond_inflow'] = bond_sales_daily['Sale_Amount']
        
        # Xử lý dữ liệu operating expenses (cash outflow)
        operating_expenses['Expense_Date'] = pd.to_datetime(operating_expenses['Expense_Date'])
        operating_expenses['date'] = operating_expenses['Expense_Date'].dt.date
        expenses_daily = operating_expenses.groupby('date').agg({
            'Expense_Amount': 'sum'
        }).reset_index()
        expenses_daily['expense_outflow'] = -expenses_daily['Expense_Amount']
        
        # Xử lý dữ liệu interbank transfers
        interbank_transfers['Transfer_Date'] = pd.to_datetime(interbank_transfers['Transfer_Date'])
        interbank_transfers['date'] = interbank_transfers['Transfer_Date'].dt.date
        
        # Tách inbound và outbound transfers
        inbound_transfers = interbank_transfers[interbank_transfers['Transfer_Type'] == 'Inbound']
        outbound_transfers = interbank_transfers[interbank_transfers['Transfer_Type'] == 'Outbound']
        
        inbound_daily = inbound_transfers.groupby('date')['Transfer_Amount'].sum().reset_index()
        inbound_daily['inbound_transfer'] = inbound_daily['Transfer_Amount']
        
        outbound_daily = outbound_transfers.groupby('date')['Transfer_Amount'].sum().reset_index()
        outbound_daily['outbound_transfer'] = -outbound_daily['Transfer_Amount']
        
        # Tạo range ngày đầy đủ
        all_dates = pd.date_range(
            start=min(deposits['Deposit_Date'].min(), loans['Loan_Date'].min(), 
                     withdrawals['Withdrawal_Date'].min(), bond_sales['Bond_Sale_Date'].min(),
                     operating_expenses['Expense_Date'].min(), interbank_transfers['Transfer_Date'].min()),
            end=max(deposits['Deposit_Date'].max(), loans['Loan_Date'].max(),
                   withdrawals['Withdrawal_Date'].max(), bond_sales['Bond_Sale_Date'].max(),
                   operating_expenses['Expense_Date'].max(), interbank_transfers['Transfer_Date'].max()),
            freq='D'
        ).date
        
        # Tạo DataFrame chính
        cash_flow_df = pd.DataFrame({'date': all_dates})
        
        # Merge tất cả dữ liệu
        cash_flow_df = cash_flow_df.merge(deposits_daily[['date', 'net_deposit_inflow']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(loans_daily[['date', 'net_loan_outflow']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(withdrawals_daily[['date', 'withdrawal_outflow']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(bond_sales_daily[['date', 'bond_inflow']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(expenses_daily[['date', 'expense_outflow']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(inbound_daily[['date', 'inbound_transfer']], on='date', how='left')
        cash_flow_df = cash_flow_df.merge(outbound_daily[['date', 'outbound_transfer']], on='date', how='left')
        
        # Điền NaN bằng 0
        cash_flow_df = cash_flow_df.fillna(0)
        
        # Tính tổng dòng tiền ròng
        cash_flow_df['net_cash_flow'] = (
            cash_flow_df['net_deposit_inflow'] + 
            cash_flow_df['net_loan_outflow'] + 
            cash_flow_df['withdrawal_outflow'] + 
            cash_flow_df['bond_inflow'] + 
            cash_flow_df['expense_outflow'] + 
            cash_flow_df['inbound_transfer'] + 
            cash_flow_df['outbound_transfer']
        )
        
        # Tính dòng tiền tích lũy
        cash_flow_df['cumulative_cash_flow'] = cash_flow_df['net_cash_flow'].cumsum()
        
        # Thêm các đặc trưng thời gian
        cash_flow_df['date'] = pd.to_datetime(cash_flow_df['date'])
        cash_flow_df['day_of_week'] = cash_flow_df['date'].dt.dayofweek
        cash_flow_df['month'] = cash_flow_df['date'].dt.month
        cash_flow_df['quarter'] = cash_flow_df['date'].dt.quarter
        cash_flow_df['day_of_month'] = cash_flow_df['date'].dt.day
        cash_flow_df['is_weekend'] = (cash_flow_df['day_of_week'] >= 5).astype(int)
        
        # Thêm moving averages
        for window in [7, 14, 30]:
            cash_flow_df[f'ma_{window}'] = cash_flow_df['net_cash_flow'].rolling(window=window).mean()
            cash_flow_df[f'std_{window}'] = cash_flow_df['net_cash_flow'].rolling(window=window).std()
        
        # Điền NaN cho moving averages
        cash_flow_df = cash_flow_df.fillna(method='bfill').fillna(0)
        
        self.cash_flow_data = cash_flow_df
        print(f"\nDữ liệu đã được xử lý: {len(cash_flow_df)} ngày")
        print(f"Từ {cash_flow_df['date'].min()} đến {cash_flow_df['date'].max()}")
        
        return cash_flow_df
    
    def create_sequences(self, data, target_column, feature_columns, sequence_length):
        """
        Tạo sequences cho LSTM
        
        Args:
            data (pd.DataFrame): Dữ liệu đầu vào
            target_column (str): Tên cột mục tiêu
            feature_columns (list): Danh sách tên cột đặc trưng
            sequence_length (int): Độ dài sequence
            
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[feature_columns].iloc[i-sequence_length:i].values)
            y.append(data[target_column].iloc[i])
            
        return np.array(X), np.array(y)
    
    def prepare_training_data(self):
        """
        Chuẩn bị dữ liệu cho việc training
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("Đang chuẩn bị dữ liệu training...")
        
        # Chọn đặc trưng cho model
        feature_columns = [
            'net_deposit_inflow', 'net_loan_outflow', 'withdrawal_outflow',
            'bond_inflow', 'expense_outflow', 'inbound_transfer', 'outbound_transfer',
            'day_of_week', 'month', 'quarter', 'day_of_month', 'is_weekend',
            'ma_7', 'ma_14', 'ma_30', 'std_7', 'std_14', 'std_30'
        ]
        
        target_column = 'net_cash_flow'
        
        # Chuẩn hóa dữ liệu
        scaled_data = self.cash_flow_data.copy()
        scaled_data[feature_columns] = self.scaler.fit_transform(scaled_data[feature_columns])
        
        # Tạo sequences
        X, y = self.create_sequences(scaled_data, target_column, feature_columns, self.sequence_length)
        
        print(f"Đã tạo {len(X)} sequences với độ dài {self.sequence_length}")
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"Training set: {len(X_train)} sequences")
        print(f"Test set: {len(X_test)} sequences")
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape):
        """
        Xây dựng mô hình LSTM
        
        Args:
            input_shape (tuple): Kích thước đầu vào
            
        Returns:
            tf.keras.Model: Mô hình LSTM
        """
        model = Sequential([
            # LSTM Layer 1
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # LSTM Layer 2
            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # LSTM Layer 3
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense Layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """
        Training mô hình LSTM
        
        Args:
            X_train, X_test, y_train, y_test: Dữ liệu training và test
            
        Returns:
            dict: Lịch sử training
        """
        print("Đang training mô hình LSTM...")
        
        # Xây dựng mô hình
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        print(f"Kiến trúc mô hình:")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
        
        # Training
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training hoàn thành!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Đánh giá mô hình
        
        Args:
            X_test, y_test: Dữ liệu test
            
        Returns:
            dict: Các metrics đánh giá
        """
        print("Đang đánh giá mô hình...")
        
        # Dự đoán
        y_pred = self.model.predict(X_test)
        
        # Tính metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Tính MAPE
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / (y_test + 1e-8))) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
        
        print("\nKết quả đánh giá:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred
    
    def predict_future(self, days_ahead=30):
        """
        Dự đoán dòng tiền trong tương lai
        
        Args:
            days_ahead (int): Số ngày cần dự đoán
            
        Returns:
            pd.DataFrame: Dự đoán dòng tiền
        """
        print(f"Đang dự đoán dòng tiền cho {days_ahead} ngày tới...")
        
        feature_columns = [
            'net_deposit_inflow', 'net_loan_outflow', 'withdrawal_outflow',
            'bond_inflow', 'expense_outflow', 'inbound_transfer', 'outbound_transfer',
            'day_of_week', 'month', 'quarter', 'day_of_month', 'is_weekend',
            'ma_7', 'ma_14', 'ma_30', 'std_7', 'std_14', 'std_30'
        ]
        
        # Lấy dữ liệu cuối cùng
        last_sequence = self.cash_flow_data[feature_columns].tail(self.sequence_length).copy()
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        # Dự đoán từng ngày
        for i in range(days_ahead):
            # Dự đoán ngày tiếp theo
            pred_input = current_sequence.reshape(1, self.sequence_length, len(feature_columns))
            next_pred = self.model.predict(pred_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Tạo đặc trưng cho ngày tiếp theo
            last_date = self.cash_flow_data['date'].iloc[-1] + timedelta(days=i+1)
            
            # Đặc trưng thời gian
            day_of_week = last_date.weekday()
            month = last_date.month
            quarter = (month - 1) // 3 + 1
            day_of_month = last_date.day
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Sử dụng trung bình của các đặc trưng giao dịch
            avg_features = current_sequence[-7:, :7].mean(axis=0)  # 7 đặc trưng giao dịch đầu tiên
            
            # Tính moving averages (simplified)
            recent_preds = predictions[-7:] if len(predictions) >= 7 else predictions
            ma_7 = np.mean(recent_preds) if recent_preds else 0
            ma_14 = ma_7  # Simplified
            ma_30 = ma_7  # Simplified
            std_7 = np.std(recent_preds) if len(recent_preds) > 1 else 0
            std_14 = std_7
            std_30 = std_7
            
            # Tạo vector đặc trưng mới
            new_features = np.concatenate([
                avg_features,
                [day_of_week, month, quarter, day_of_month, is_weekend,
                 ma_7, ma_14, ma_30, std_7, std_14, std_30]
            ])
            
            # Cập nhật sequence
            current_sequence = np.vstack([current_sequence[1:], new_features])
        
        # Tạo DataFrame kết quả
        future_dates = pd.date_range(
            start=self.cash_flow_data['date'].iloc[-1] + timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_cash_flow': predictions
        })
        
        # Tính dòng tiền tích lũy
        last_cumulative = self.cash_flow_data['cumulative_cash_flow'].iloc[-1]
        future_df['predicted_cumulative_cash_flow'] = last_cumulative + np.cumsum(predictions)
        
        return future_df
    
    def plot_results(self, history, y_test, y_pred, future_predictions):
        """
        Vẽ biểu đồ kết quả
        
        Args:
            history: Lịch sử training
            y_test, y_pred: Dữ liệu test và dự đoán
            future_predictions: Dự đoán tương lai
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Training history
        axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Model Training History', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted
        axes[0, 1].scatter(y_test, y_pred, alpha=0.6, color='green')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_title('Actual vs Predicted Cash Flow', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Actual Cash Flow')
        axes[0, 1].set_ylabel('Predicted Cash Flow')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Historical Cash Flow
        axes[1, 0].plot(self.cash_flow_data['date'], self.cash_flow_data['net_cash_flow'], 
                       label='Historical Cash Flow', color='blue', alpha=0.7)
        axes[1, 0].plot(self.cash_flow_data['date'], self.cash_flow_data['cumulative_cash_flow'], 
                       label='Cumulative Cash Flow', color='orange', alpha=0.7)
        axes[1, 0].set_title('Historical Cash Flow Trends', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Cash Flow')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Future Predictions
        axes[1, 1].plot(future_predictions['date'], future_predictions['predicted_cash_flow'], 
                       label='Predicted Daily Cash Flow', color='red', marker='o', markersize=4)
        axes[1, 1].plot(future_predictions['date'], future_predictions['predicted_cumulative_cash_flow'], 
                       label='Predicted Cumulative Cash Flow', color='purple', alpha=0.7)
        axes[1, 1].set_title('Future Cash Flow Predictions', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cash Flow')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/cash_flow_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, metrics, future_predictions):
        """
        Lưu kết quả phân tích
        
        Args:
            metrics (dict): Các metrics đánh giá
            future_predictions (pd.DataFrame): Dự đoán tương lai
        """
        # Lưu metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.data_path}/model_evaluation_metrics.csv', index=False)
        
        # Lưu dự đoán tương lai
        future_predictions.to_csv(f'{self.data_path}/future_cash_flow_predictions.csv', index=False)
        
        # Lưu dữ liệu dòng tiền đã xử lý
        self.cash_flow_data.to_csv(f'{self.data_path}/processed_cash_flow_data.csv', index=False)
        
        print(f"\nKết quả đã được lưu tại:")
        print(f"  - Model metrics: {self.data_path}/model_evaluation_metrics.csv")
        print(f"  - Future predictions: {self.data_path}/future_cash_flow_predictions.csv")
        print(f"  - Processed data: {self.data_path}/processed_cash_flow_data.csv")
        print(f"  - Analysis charts: {self.data_path}/cash_flow_analysis_results.png")
    
    def run_complete_pipeline(self, days_ahead=30):
        """
        Chạy pipeline hoàn chỉnh
        
        Args:
            days_ahead (int): Số ngày cần dự đoán
            
        Returns:
            dict: Kết quả pipeline
        """
        print("=" * 60)
        print("CASH FLOW FORECASTING PIPELINE USING LSTM")
        print("=" * 60)
        
        # Bước 1: Tải và xử lý dữ liệu
        cash_flow_df = self.load_and_preprocess_data()
        
        # Bước 2: Chuẩn bị dữ liệu training
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        # Bước 3: Training mô hình
        history = self.train_model(X_train, X_test, y_train, y_test)
        
        # Bước 4: Đánh giá mô hình
        metrics, y_pred = self.evaluate_model(X_test, y_test)
        
        # Bước 5: Dự đoán tương lai
        future_predictions = self.predict_future(days_ahead)
        
        # Bước 6: Vẽ biểu đồ kết quả
        self.plot_results(history, y_test, y_pred, future_predictions)
        
        # Bước 7: Lưu kết quả
        self.save_results(metrics, future_predictions)
        
        print("\n" + "=" * 60)
        print("PIPELINE HOÀN THÀNH THÀNH CÔNG!")
        print("=" * 60)
        
        return {
            'metrics': metrics,
            'future_predictions': future_predictions,
            'cash_flow_data': cash_flow_df,
            'model': self.model
        }

def main():
    """
    Hàm chính để chạy pipeline
    """
    # Đường dẫn đến dữ liệu
    data_path = "C:/Users/Than Thien/PycharmProjects/data_analysis_IT7327_K15.2/UC_4"
    
    # Tạo và chạy pipeline
    pipeline = CashFlowLSTMPipeline(data_path)
    results = pipeline.run_complete_pipeline(days_ahead=30)
    
    # In tóm tắt kết quả
    print(f"\nTÓM TẮT KẾT QUẢ:")
    print(f"- Model R²: {results['metrics']['R²']:.4f}")
    print(f"- MAPE: {results['metrics']['MAPE']:.2f}%")
    print(f"- RMSE: {results['metrics']['RMSE']:.2f}")
    
    # Hiển thị một số dự đoán
    print(f"\nDỰ ĐOÁN DÒNG TIỀN CHO 7 NGÀY TỚI:")
    future_sample = results['future_predictions'].head(7)
    for _, row in future_sample.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['predicted_cash_flow']:,.0f}")

if __name__ == "__main__":
    main()