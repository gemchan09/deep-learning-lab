"""
BT02 - Phân tích khám phá và tiền xử lý dữ liệu
File: data_exploration.py

Phạm vi:
- Câu 1: Data Overview
- Câu 2: Descriptive Statistics
- Câu 3: Histogram (Tuổi), Boxplot (Lương), phân tích phân phối
- Câu 4: Scatter plot (Tuổi - Lương), xu hướng và dữ liệu bất thường

Cài đặt thư viện:
    pip install pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_dataset():
    """Tạo DataFrame đúng theo bảng dữ liệu trong đề BT02."""
    data = {
        "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Tuổi": [25, 30, 22, 40, 35, np.nan, 29, 50, 27, 31],
        "Lương": [8000, np.nan, 5000, 15000, -2000, 7000, 8500, 30000, 8200, 7800],
        "Điểm": [7.5, 8.0, 6.5, 9.0, 7.0, 6.0, 8.5, 9.5, np.nan, 7.8],
        "Giới tính": ["Nam", "Nữ", "Nam", "Nữ", "Nam", "Nữ", "Nam", "Nữ", "Nam", "Nữ"],
        "Mua hàng": ["Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes"],
    }
    return pd.DataFrame(data)


def data_overview(df):
    """In số lượng mẫu, thuộc tính và phân loại kiểu dữ liệu."""
    print("\n" + "=" * 70)
    print("CÂU 1 - DATA OVERVIEW")
    print("=" * 70)
    print(f"Số lượng mẫu (n): {len(df)}")
    print(f"Tổng số cột, bao gồm ID: {len(df.columns)}")
    print(f"Số thuộc tính không tính ID: {len(df.columns) - 1}")
    print("\nBảng dữ liệu gốc:")
    print(df.to_string(index=False))

    numerical = ["Tuổi", "Lương", "Điểm"]
    categorical = ["Giới tính", "Mua hàng"]
    identifier = ["ID"]

    print("\nPhân loại thuộc tính:")
    print(f"- Numerical (định lượng): {', '.join(numerical)}")
    print(f"- Categorical (định tính): {', '.join(categorical)}")
    print(f"- Identifier (mã định danh): {', '.join(identifier)}")
    print("Lưu ý: ID được biểu diễn bằng số nhưng là mã định danh,")
    print("không nên dùng như một biến định lượng để tính thống kê.")


def descriptive_statistics(df):
    """Tính Mean, Median, Mode cho các thuộc tính số."""
    print("\n" + "=" * 70)
    print("CÂU 2 - DESCRIPTIVE STATISTICS")
    print("=" * 70)

    columns = ["Tuổi", "Lương", "Điểm"]
    rows = []

    for column in columns:
        values = df[column].dropna()
        mean_value = values.mean()
        median_value = values.median()
        modes = values.mode().tolist()
        mode_text = ", ".join(f"{x:g}" for x in modes) if modes else "Không có"

        rows.append({
            "Thuộc tính": column,
            "Số giá trị hợp lệ": len(values),
            "Mean": round(mean_value, 4),
            "Median": round(median_value, 4),
            "Mode": mode_text if len(modes) < len(values) else "Không có mode duy nhất",
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    print("\nNhận xét:")
    print("- Mean của Lương có thể bị ảnh hưởng bởi các giá trị cực đoan.")
    print("- Lương = -2000 là giá trị cần kiểm tra về chất lượng dữ liệu.")
    print("- Median thường ít nhạy với outlier hơn Mean.")
    print("Lưu ý: thống kê ở đây được tính trên các giá trị không thiếu,")
    print("và chưa tự động sửa hoặc loại bỏ giá trị bất thường.")


def distribution_analysis(df, output_dir="."):
    """Vẽ histogram tuổi, boxplot lương và in nhận xét."""
    print("\n" + "=" * 70)
    print("CÂU 3 - DISTRIBUTION ANALYSIS")
    print("=" * 70)

    ages = df["Tuổi"].dropna()
    salaries = df["Lương"].dropna()

    age_mean = ages.mean()
    age_median = ages.median()
    salary_mean = salaries.mean()
    salary_median = salaries.median()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(ages, bins="auto", edgecolor="black")
    axes[0].set_title("Histogram - Tuổi")
    axes[0].set_xlabel("Tuổi")
    axes[0].set_ylabel("Tần số")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(salaries, vert=True)
    axes[1].set_title("Boxplot - Lương")
    axes[1].set_ylabel("Lương")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = os.path.join(output_dir, "distribution_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {path}")
    print(f"Tuổi: Mean = {age_mean:.2f}, Median = {age_median:.2f}.")
    if age_mean > age_median:
        print("Nhận xét sơ bộ: Mean > Median, có thể gợi ý phân phối lệch phải.")
    elif age_mean < age_median:
        print("Nhận xét sơ bộ: Mean < Median, có thể gợi ý phân phối lệch trái.")
    else:
        print("Mean bằng Median; cần xem biểu đồ để đánh giá hình dạng phân phối.")

    print(f"Lương: Mean = {salary_mean:.2f}, Median = {salary_median:.2f}.")
    if salary_mean > salary_median:
        print("Nhận xét sơ bộ: Mean > Median, có thể gợi ý phân phối lệch phải.")
    elif salary_mean < salary_median:
        print("Nhận xét sơ bộ: Mean < Median, có thể gợi ý phân phối lệch trái.")

    print("Dữ liệu cần kiểm tra: Lương = -2000 (không hợp lý theo ngữ cảnh lương)")
    print("và Lương = 30000 (giá trị cao, cần kiểm tra bằng phương pháp IQR ở Câu 8).")


def relationship_analysis(df, output_dir="."):
    """Vẽ scatter plot tuổi-lương trên các dòng có đủ cả hai giá trị."""
    print("\n" + "=" * 70)
    print("CÂU 4 - RELATIONSHIP ANALYSIS")
    print("=" * 70)

    paired = df.dropna(subset=["Tuổi", "Lương"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(paired["Tuổi"], paired["Lương"])

    for _, row in paired.iterrows():
        ax.annotate(str(int(row["ID"])), (row["Tuổi"], row["Lương"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_title("Scatter plot - Tuổi và Lương")
    ax.set_xlabel("Tuổi")
    ax.set_ylabel("Lương")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "age_salary_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {path}")
    if len(paired) >= 2:
        correlation = paired["Tuổi"].corr(paired["Lương"])
        print(f"Hệ số tương quan Pearson trên các cặp không thiếu: {correlation:.4f}")
        print("Hệ số tương quan chỉ mô tả quan hệ tuyến tính trên dữ liệu hiện có;")
        print("cần thận trọng vì có lương âm và một giá trị lương rất cao.")
    print("Các điểm cần kiểm tra:")
    print("- ID 5: Tuổi = 35, Lương = -2000 (giá trị lương bất thường).")
    print("- ID 8: Tuổi = 50, Lương = 30000 (giá trị lương cao, cần kiểm tra outlier).")
    print("Đề xuất: xác minh dữ liệu gốc, xử lý missing values, xử lý lương âm")
    print("theo thông tin nghiệp vụ, sau đó mới scaling và huấn luyện mô hình.")


def main():
    df = create_dataset()
    data_overview(df)
    descriptive_statistics(df)
    distribution_analysis(df)
    relationship_analysis(df)
    print("\nHoàn tất phân tích khám phá dữ liệu.")


if __name__ == "__main__":
    main()
