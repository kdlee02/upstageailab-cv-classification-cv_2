"""
sample_submission.csv 순서대로 예측 결과를 저장하는 기능 테스트
"""

import pandas as pd
from src.utils.prediction import save_predictions_in_sample_order

def test_sample_order_save():
    """sample_submission 순서대로 저장하는 기능 테스트"""
    
    print("=== sample_submission 순서 저장 테스트 ===")
    
    # 예시 예측 결과 (실제로는 모델에서 예측한 결과)
    example_filenames = [
        "0008fdb22ddce0ce.jpg",
        "00091bffdffd83de.jpg", 
        "00396fbc1f6cc21d.jpg",
        "00471f8038d9c4b6.jpg",
        "00901f504008d884.jpg"
    ]
    
    example_predictions = [1, 2, 0, 3, 1]  # 예시 예측 클래스
    
    # sample_submission 순서대로 저장
    output_path = "test_predictions_sample_order.csv"
    
    result_df = save_predictions_in_sample_order(
        filenames=example_filenames,
        predictions=example_predictions,
        output_path=output_path,
        sample_submission_path="data/sample_submission.csv"
    )
    
    print("\n저장된 결과 미리보기:")
    print(result_df.head(10))
    
    # sample_submission.csv와 비교
    sample_df = pd.read_csv("data/sample_submission.csv")
    
    print(f"\n=== 검증 결과 ===")
    print(f"sample_submission.csv 행 수: {len(sample_df)}")
    print(f"저장된 결과 행 수: {len(result_df)}")
    
    # ID 순서 비교
    if (sample_df['ID'] == result_df['ID']).all():
        print("✅ ID 순서가 sample_submission.csv와 일치합니다!")
    else:
        print("❌ ID 순서가 sample_submission.csv와 다릅니다.")
    
    # 예측된 ID들의 값 확인
    print(f"\n=== 예측된 ID들의 값 ===")
    for filename in example_filenames:
        if filename in result_df['ID'].values:
            pred_value = result_df[result_df['ID'] == filename]['target'].iloc[0]
            print(f"{filename}: {pred_value}")
    
    print(f"\n=== 테스트 완료 ===")
    print(f"결과 파일: {output_path}")
    
    return result_df

if __name__ == "__main__":
    test_sample_order_save() 