#!/usr/bin/env python3
"""
환경 변수 사용 예시 스크립트
"""

import os
from src.utils.env_utils import (
    load_env_variables, 
    get_s3_config, 
    get_wandb_config, 
    is_s3_enabled,
    check_required_env_vars
)
from src.utils.s3_utils import create_s3_handler


def example_env_loading():
    """환경 변수 로딩 예시"""
    print("=== 환경 변수 로딩 예시 ===")
    
    # 환경 변수 로드
    load_env_variables()
    
    # S3 설정 확인
    s3_config = get_s3_config()
    print(f"S3 설정:")
    print(f"  - 활성화: {s3_config['enabled']}")
    print(f"  - 버킷 이름: {s3_config['bucket_name']}")
    print(f"  - 리전: {s3_config['region_name']}")
    
    # W&B 설정 확인
    wandb_config = get_wandb_config()
    print(f"W&B 설정:")
    print(f"  - 프로젝트: {wandb_config['project']}")
    print(f"  - API 키 설정: {'예' if wandb_config['api_key'] else '아니오'}")


def example_s3_handler():
    """S3 핸들러 사용 예시"""
    print("\n=== S3 핸들러 사용 예시 ===")
    
    # S3가 활성화되어 있는지 확인
    if is_s3_enabled():
        print("S3가 활성화되어 있습니다.")
        
        # S3 핸들러 생성
        s3_handler = create_s3_handler()
        if s3_handler:
            print(f"S3 핸들러 생성 성공: {s3_handler.bucket_name}")
            
            # 파일 목록 조회 예시
            files = s3_handler.list_files(prefix="models/")
            print(f"모델 파일 수: {len(files)}")
            
            if files:
                print("모델 파일들:")
                for file in files[:5]:  # 처음 5개만 출력
                    print(f"  - {file}")
        else:
            print("S3 핸들러 생성에 실패했습니다.")
    else:
        print("S3가 활성화되지 않았습니다.")
        print(".env 파일에서 S3_BUCKET_NAME을 설정하세요.")


def example_required_vars_check():
    """필수 환경 변수 확인 예시"""
    print("\n=== 필수 환경 변수 확인 예시 ===")
    
    # 필수 환경 변수 확인
    is_valid = check_required_env_vars()
    
    if is_valid:
        print("모든 필수 환경 변수가 설정되어 있습니다.")
    else:
        print("일부 필수 환경 변수가 설정되지 않았습니다.")
        print("다음 변수들을 .env 파일에 설정하세요:")
        print("  - WANDB_API_KEY")
        print("  - WANDB_PROJECT")


def example_env_file_creation():
    """환경 변수 파일 생성 가이드"""
    print("\n=== 환경 변수 파일 생성 가이드 ===")
    
    env_example_path = "env_example.txt"
    env_path = ".env"
    
    if os.path.exists(env_example_path):
        print(f"1. {env_example_path} 파일을 {env_path}로 복사하세요:")
        print(f"   cp {env_example_path} {env_path}")
        print()
        print("2. .env 파일을 편집하여 실제 값들을 설정하세요:")
        print("   - WANDB_API_KEY: Weights & Biases API 키")
        print("   - WANDB_PROJECT: 프로젝트 이름")
        print("   - S3_BUCKET_NAME: S3 버킷 이름 (선택사항)")
        print("   - AWS_ACCESS_KEY_ID: AWS 액세스 키 (선택사항)")
        print("   - AWS_SECRET_ACCESS_KEY: AWS 시크릿 키 (선택사항)")
        print("   - AWS_DEFAULT_REGION: AWS 리전 (기본값: us-east-1)")
    else:
        print(f"{env_example_path} 파일을 찾을 수 없습니다.")


def example_manual_s3_config():
    """수동 S3 설정 예시"""
    print("\n=== 수동 S3 설정 예시 ===")
    
    # 환경 변수에서 S3 설정 가져오기
    s3_config = get_s3_config()
    
    if s3_config['enabled']:
        print("환경 변수에서 S3 설정을 가져왔습니다:")
        print(f"  버킷: {s3_config['bucket_name']}")
        print(f"  리전: {s3_config['region_name']}")
        
        # 수동으로 S3 핸들러 생성
        try:
            from src.utils.s3_utils import S3Handler
            s3_handler = S3Handler(
                bucket_name=s3_config['bucket_name'],
                region_name=s3_config['region_name']
            )
            print("S3 핸들러 생성 성공!")
        except Exception as e:
            print(f"S3 핸들러 생성 실패: {e}")
    else:
        print("S3 설정이 없습니다. .env 파일에서 S3_BUCKET_NAME을 설정하세요.")


if __name__ == "__main__":
    print("환경 변수 사용 예시를 실행합니다...")
    print("=" * 50)
    
    try:
        example_env_loading()
        example_s3_handler()
        example_required_vars_check()
        example_env_file_creation()
        example_manual_s3_config()
        
        print("\n" + "=" * 50)
        print("모든 예시가 완료되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("환경 변수 설정을 확인하세요.") 