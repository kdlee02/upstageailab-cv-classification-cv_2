#!/usr/bin/env python3
"""
스캐줄러 사용 예제 스크립트
다양한 스캐줄러를 사용하여 실험을 실행하는 방법을 보여줍니다.
"""

import os
import sys
import subprocess
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_experiment_with_scheduler(scheduler_name: str, experiment_name: str | None = None):
    """지정된 스캐줄러로 실험을 실행합니다."""
    
    if experiment_name is None:
        experiment_name = f"experiment_{scheduler_name}"
    
    print(f"=== {scheduler_name.upper()} 스캐줄러로 실험 실행 ===")
    print(f"실험 이름: {experiment_name}")
    
    # 명령어 구성
    cmd = [
        "python", "run_experiment.py",
        f"scheduler={scheduler_name}",
        f"experiment.name={experiment_name}",
        f"experiment.description=스캐줄러 테스트: {scheduler_name}",
        f"experiment.tags=[scheduler_test, {scheduler_name}]"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 실험 실행
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("실험 완료!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"실험 실행 중 오류 발생: {e}")
        print(f"오류 출력: {e.stderr}")
        return False
    
    return True

def main():
    """메인 함수 - 다양한 스캐줄러로 실험을 실행합니다."""
    
    print("스캐줄러 사용 예제")
    print("=" * 50)
    
    # 지원하는 스캐줄러 목록
    schedulers = [
        "cosine",
        "step", 
        "exponential",
        "plateau",
        "warmup_cosine",
        "warmup_linear",
        "cosine_warm_restart"
    ]
    
    print("지원하는 스캐줄러:")
    for i, scheduler in enumerate(schedulers, 1):
        print(f"  {i}. {scheduler}")
    
    print("\n사용 방법:")
    print("1. 개별 스캐줄러로 실험 실행")
    print("2. 모든 스캐줄러로 비교 실험 실행")
    print("3. 종료")
    
    while True:
        choice = input("\n선택하세요 (1-3): ").strip()
        
        if choice == "1":
            print("\n개별 스캐줄러 선택:")
            for i, scheduler in enumerate(schedulers, 1):
                print(f"  {i}. {scheduler}")
            
            try:
                scheduler_idx = int(input("스캐줄러 번호를 선택하세요: ")) - 1
                if 0 <= scheduler_idx < len(schedulers):
                    scheduler_name = schedulers[scheduler_idx]
                    experiment_name = input(f"실험 이름을 입력하세요 (기본값: experiment_{scheduler_name}): ").strip()
                    if not experiment_name:
                        experiment_name = None
                    
                    run_experiment_with_scheduler(scheduler_name, experiment_name)
                else:
                    print("잘못된 번호입니다.")
            except ValueError:
                print("숫자를 입력해주세요.")
                
        elif choice == "2":
            print("\n모든 스캐줄러로 비교 실험을 실행합니다...")
            base_name = input("기본 실험 이름을 입력하세요 (기본값: scheduler_comparison): ").strip()
            if not base_name:
                base_name = "scheduler_comparison"
            
            for scheduler in schedulers:
                experiment_name = f"{base_name}_{scheduler}"
                success = run_experiment_with_scheduler(scheduler, experiment_name)
                if not success:
                    print(f"{scheduler} 스캐줄러 실험 실패. 다음 스캐줄러로 진행합니다.")
                print("\n" + "="*50 + "\n")
                
        elif choice == "3":
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("1-3 중에서 선택해주세요.")

if __name__ == "__main__":
    main() 