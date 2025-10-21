# Cash Flow Forecasting Pipeline with LSTM 🚀

Pipeline hoàn chỉnh để dự đoán dòng tiền của ngân hàng sử dụng mô hình LSTM (Long Short-Term Memory). Pipeline này tích hợp dữ liệu từ 6 nguồn khác nhau để tạo ra dự đoán dòng tiền chính xác và đáng tin cậy.

## 📋 Mục lục
- [Tổng quan](#-tổng-quan)
- [Dữ liệu đầu vào](#-dữ-liệu-đầu-vào)
- [Cài đặt](#-cài-đặt)
- [Cách sử dụng](#-cách-sử-dụng)
- [Kiến trúc Pipeline](#-kiến-trúc-pipeline)
- [Kết quả đầu ra](#-kết-quả-đầu-ra)
- [Tùy chỉnh](#-tùy-chỉnh)
- [Troubleshooting](#-troubleshooting)

## 🎯 Tổng quan

Pipeline này được thiết kế để giải quyết bài toán dự đoán dòng tiền trong ngành ngân hàng, giúp:

- **Quản lý thanh khoản**: Dự đoán nhu cầu tiền mặt hàng ngày
- **Cảnh báo rủi ro**: Phát hiện sớm những ngày có thể thiếu hụt dòng tiền
- **Tối ưu hóa vốn**: Hỗ trợ quyết định đầu tư và cho vay
- **Tuân thủ quy định**: Đảm bảo các tỷ lệ thanh khoản theo yêu cầu

### 🏆 Đặc điểm nổi bật

- ✅ **Tích hợp đa nguồn**: Xử lý 6 loại dữ liệu giao dịch khác nhau
- ✅ **Model LSTM tiên tiến**: Sử dụng deep learning với regularization
- ✅ **Features Engineering**: Tự động tạo đặc trưng thời gian và statistical
- ✅ **Đánh giá toàn diện**: Multiple metrics và visualization
- ✅ **Dự đoán linh hoạt**: Có thể dự đoán từ 1 đến nhiều tháng
- ✅ **Tự động hóa hoàn toàn**: Chỉ cần 1 lệnh để chạy toàn bộ pipeline

## 📊 Dữ liệu đầu vào

Pipeline yêu cầu 6 file CSV sau:

### 1. `fact_deposits.csv` - Dữ liệu tiền gửi
```
Deposit_ID, Customer_ID, Deposit_Date, Deposit_Amount, Account_Type, Term, Interest_Rate, Interest_Outflow
```

### 2. `fact_loans.csv` - Dữ liệu cho vay
```
Loan_ID, Customer_ID, Loan_Date, Loan_Amount, Loan_Type, Loan_Term, Interest_Rate, Interest_Inflow
```

### 3. `fact_withdrawals.csv` - Dữ liệu rút tiền
```
Withdrawal_Date, Customer_ID, Withdrawal_Amount, Account_Type, Withdrawal_Channel, Branch/ATM_ID
```

### 4. `fact_bond_sales.csv` - Dữ liệu bán trái phiếu
```
Bond_Sale_Date, Bond_ID, Sale_Amount, Bond_Maturity_Date, Bond_Type, Interest_Rate
```

### 5. `fact_operating_expenses.csv` - Chi phí vận hành
```
Expense_Date, Expense_Type, Expense_Amount, Payment_Method, Cost_Center/Department
```

### 6. `fact_interbank_transfers.csv` - Chuyển khoản liên ngân hàng
```
Transfer_Date, Transaction_ID, Counterparty_Bank, Transfer_Amount, Transfer_Currency, Transfer_Type, Transfer_Purpose
```

## 🛠️ Cài đặt

### Bước 1: Clone repository hoặc tải file
```bash
# Tải các file pipeline vào thư mục của bạn
```

### Bước 2: Cài đặt Python dependencies
```bash
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu
Đảm bảo tất cả 6 file CSV nằm trong cùng thư mục với các file pipeline.

## 🚀 Cách sử dụng

### Phương pháp 1: Chạy trực tiếp từ command line (Khuyến nghị)

```bash
# Dự đoán 30 ngày với dữ liệu ở thư mục hiện tại
python quick_start.py

# Dự đoán 60 ngày
python quick_start.py --days 60

# Chỉ định đường dẫn dữ liệu khác
python quick_start.py --days 30 --data_path "./data"

# Chạy ở chế độ im lặng
python quick_start.py --quiet
```

### Phương pháp 2: Sử dụng Jupyter Notebook

```bash
# Mở notebook demo
jupyter notebook cash_flow_demo.ipynb
```

Notebook này cung cấp giao diện trực quan và hướng dẫn từng bước.

### Phương pháp 3: Import trong Python code

```python
from cash_flow_lstm_pipeline import CashFlowLSTMPipeline

# Khởi tạo pipeline
pipeline = CashFlowLSTMPipeline(data_path="./")

# Chạy pipeline hoàn chỉnh
results = pipeline.run_complete_pipeline(days_ahead=30)

# Sử dụng kết quả
print(f"Model R²: {results['metrics']['R²']:.4f}")
```

## 🏗️ Kiến trúc Pipeline

Pipeline bao gồm 7 bước chính:

### 1. **Data Loading & Preprocessing** 📥
- Tải dữ liệu từ 6 file CSV
- Xử lý định dạng ngày tháng
- Tính toán dòng tiền ròng theo từng nguồn
- Tổng hợp dữ liệu theo ngày

### 2. **Feature Engineering** 🔧
- Đặc trưng thời gian: ngày trong tuần, tháng, quý
- Moving averages: 7, 14, 30 ngày
- Standard deviation windows
- Dòng tiền tích lũy

### 3. **Data Preparation** 📋
- Chuẩn hóa dữ liệu với MinMaxScaler
- Tạo sequences với sliding window (30 ngày)
- Chia train/test set (80/20)

### 4. **Model Building** 🧠
```
LSTM Architecture:
├── LSTM Layer 1 (128 units) + BatchNorm + Dropout
├── LSTM Layer 2 (64 units) + BatchNorm + Dropout  
├── LSTM Layer 3 (32 units) + BatchNorm + Dropout
├── Dense Layer (64 units) + Dropout
├── Dense Layer (32 units)
└── Output Layer (1 unit)
```

### 5. **Training** 🎯
- Adam optimizer với learning rate scheduling
- Early stopping để tránh overfitting
- Batch size: 32, Max epochs: 100

### 6. **Evaluation** 📊
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error) 
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### 7. **Prediction & Visualization** 🔮
- Dự đoán multi-step cho số ngày mong muốn
- Tự động cập nhật features cho ngày tiếp theo
- Tạo biểu đồ phân tích và lưu kết quả

## 📄 Kết quả đầu ra

Pipeline tự động tạo ra các file sau:

### 1. `model_evaluation_metrics.csv`
Chứa các metrics đánh giá mô hình:
```csv
MAE,MSE,RMSE,R²,MAPE
1234.56,1234567.89,1111.11,0.8542,12.34
```

### 2. `future_cash_flow_predictions.csv`
Dự đoán dòng tiền tương lai:
```csv
date,predicted_cash_flow,predicted_cumulative_cash_flow
2025-10-22,150000,12500000
2025-10-23,-75000,12425000
...
```

### 3. `processed_cash_flow_data.csv`
Dữ liệu đã được xử lý và tính toán features:
```csv
date,net_deposit_inflow,net_loan_outflow,withdrawal_outflow,bond_inflow,expense_outflow,inbound_transfer,outbound_transfer,net_cash_flow,cumulative_cash_flow,day_of_week,month,quarter,day_of_month,is_weekend,ma_7,ma_14,ma_30,std_7,std_14,std_30
```

### 4. `cash_flow_analysis_results.png`
Biểu đồ tổng hợp bao gồm:
- Training history (loss curves)
- Actual vs Predicted scatter plot
- Historical cash flow trends
- Future predictions visualization

## ⚙️ Tùy chỉnh

### Thay đổi tham số mô hình

Chỉnh sửa trong class `CashFlowLSTMPipeline`:

```python
# Thay đổi sequence length
self.sequence_length = 60  # Sử dụng 60 ngày thay vì 30

# Thay đổi kiến trúc LSTM
def build_lstm_model(self, input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),  # Tăng units
        # ... thêm layers khác
    ])
```

### Thêm features mới

```python
# Trong hàm load_and_preprocess_data
# Thêm các đặc trưng mới
cash_flow_df['rolling_volatility'] = cash_flow_df['net_cash_flow'].rolling(window=30).std()
cash_flow_df['momentum'] = cash_flow_df['net_cash_flow'].pct_change()
```

### Thay đổi metrics đánh giá

```python
# Trong hàm evaluate_model
# Thêm metrics mới
from sklearn.metrics import mean_absolute_percentage_error
mape_sklearn = mean_absolute_percentage_error(y_test, y_pred)
```

## 🔧 Troubleshooting

### Lỗi thường gặp và cách khắc phục:

#### 1. **ModuleNotFoundError**
```bash
# Lỗi
ModuleNotFoundError: No module named 'tensorflow'

# Khắc phục
pip install -r requirements.txt
```

#### 2. **File không tìm thấy**
```bash
# Lỗi
FileNotFoundError: [Errno 2] No such file or directory: 'fact_deposits.csv'

# Khắc phục
# Đảm bảo tất cả 6 file CSV có trong thư mục
ls *.csv  # Linux/Mac
dir *.csv # Windows
```

#### 3. **Memory Error**
```bash
# Lỗi
MemoryError: Unable to allocate array

# Khắc phục
# Giảm batch_size trong hàm train_model
history = self.model.fit(
    X_train, y_train,
    batch_size=16,  # Giảm từ 32 xuống 16
    # ...
)
```

#### 4. **Kết quả dự đoán không tốt**
Nếu MAPE > 30% hoặc R² < 0.5:

- Tăng sequence_length (60-90 ngày)
- Thêm nhiều features engineering
- Tăng số epochs training
- Thử các kiến trúc mô hình khác

#### 5. **Dữ liệu thiếu hoặc không đầy đủ**
```python
# Kiểm tra dữ liệu
print(cash_flow_df.isnull().sum())
print(cash_flow_df.describe())
```

### Performance Tips:

1. **Sử dụng GPU**: Cài đặt `tensorflow-gpu` nếu có GPU
2. **Tối ưu bộ nhớ**: Giảm batch_size nếu bị out of memory
3. **Parallel processing**: Sử dụng `n_jobs=-1` trong các scikit-learn functions
4. **Data caching**: Lưu processed data để không phải xử lý lại

## 📞 Hỗ trợ

Nếu gặp vấn đề hoặc cần hỗ trợ:

1. Kiểm tra [Troubleshooting](#-troubleshooting) section
2. Đảm bảo dữ liệu có định dạng đúng
3. Kiểm tra log output để tìm lỗi cụ thể
4. Thử chạy với sample data nhỏ trước

## 📈 Kết quả mong đợi

Với dữ liệu chất lượng tốt, pipeline thường đạt được:

- **R² Score**: 0.7 - 0.9 (càng gần 1 càng tốt)
- **MAPE**: 10% - 25% (càng thấp càng tốt)
- **Training time**: 5-15 phút (tùy thuộc vào hardware)

---

**Happy Forecasting! 🚀📊💰**