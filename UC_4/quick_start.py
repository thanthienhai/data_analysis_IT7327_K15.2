"""
Quick Start Script for Cash Flow LSTM Pipeline
==============================================

Script đơn giản để chạy pipeline dự đoán dòng tiền từ command line.

Usage:
    python quick_start.py [--days DAYS] [--data_path PATH]

Examples:
    python quick_start.py
    python quick_start.py --days 60
    python quick_start.py --days 30 --data_path "./data"
"""

import argparse
import os
import sys
from cash_flow_lstm_pipeline import CashFlowLSTMPipeline

def main():
    """
    Hàm chính để chạy pipeline với tham số từ command line
    """
    # Thiết lập argument parser
    parser = argparse.ArgumentParser(
        description='Cash Flow Forecasting Pipeline using LSTM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Dự đoán 30 ngày với dữ liệu hiện tại
  %(prog)s --days 60                 # Dự đoán 60 ngày
  %(prog)s --days 30 --data_path ./  # Chỉ định đường dẫn dữ liệu
        """
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Số ngày cần dự đoán (mặc định: 30)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='.',
        help='Đường dẫn đến thư mục chứa dữ liệu CSV (mặc định: thư mục hiện tại)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Chạy ở chế độ im lặng (ít output)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Kiểm tra đường dẫn dữ liệu
    if not os.path.exists(args.data_path):
        print(f"❌ Lỗi: Đường dẫn {args.data_path} không tồn tại!")
        sys.exit(1)
    
    # Kiểm tra các file CSV cần thiết
    required_files = [
        'fact_deposits.csv',
        'fact_loans.csv', 
        'fact_withdrawals.csv',
        'fact_bond_sales.csv',
        'fact_operating_expenses.csv',
        'fact_interbank_transfers.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(args.data_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Lỗi: Không tìm thấy các file cần thiết:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nVui lòng đảm bảo tất cả file CSV có trong thư mục: {args.data_path}")
        sys.exit(1)
    
    # Hiển thị thông tin
    if not args.quiet:
        print("🚀 CASH FLOW FORECASTING PIPELINE")
        print("=" * 50)
        print(f"📁 Đường dẫn dữ liệu: {os.path.abspath(args.data_path)}")
        print(f"📅 Số ngày dự đoán: {args.days}")
        print(f"🤖 Mô hình: LSTM")
        print("=" * 50)
    
    try:
        # Khởi tạo và chạy pipeline
        pipeline = CashFlowLSTMPipeline(args.data_path)
        results = pipeline.run_complete_pipeline(days_ahead=args.days)
        
        # Hiển thị kết quả tóm tắt
        if not args.quiet:
            print("\n" + "=" * 50)
            print("📊 KẾT QUẢ TỔNG QUAN")
            print("=" * 50)
            
            metrics = results['metrics']
            print(f"🎯 Model R²: {metrics['R²']:.4f}")
            print(f"📈 MAPE: {metrics['MAPE']:.2f}%")
            print(f"📊 RMSE: {metrics['RMSE']:,.0f}")
            
            # Hiển thị một số dự đoán đầu tiên
            future_df = results['future_predictions']
            print(f"\n🔮 DỰ ĐOÁN {min(7, args.days)} NGÀY ĐẦU TIÊN:")
            for _, row in future_df.head(7).iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                cash_flow = row['predicted_cash_flow']
                status = "📈" if cash_flow > 0 else "📉"
                print(f"   {status} {date_str}: {cash_flow:>10,.0f}")
            
            print(f"\n💾 Kết quả đã được lưu tại: {os.path.abspath(args.data_path)}")
            print("   - model_evaluation_metrics.csv")
            print("   - future_cash_flow_predictions.csv") 
            print("   - processed_cash_flow_data.csv")
            print("   - cash_flow_analysis_results.png")
            
        else:
            # Chế độ im lặng - chỉ hiển thị metrics chính
            metrics = results['metrics']
            print(f"R²={metrics['R²']:.4f}, MAPE={metrics['MAPE']:.2f}%, RMSE={metrics['RMSE']:,.0f}")
        
        print("\n✅ Pipeline hoàn thành thành công!")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi chạy pipeline: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()