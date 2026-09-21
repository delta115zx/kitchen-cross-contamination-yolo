# 프로젝트 구조 리뷰 체크리스트

"프로젝트를 구조적으로 하나하나 뜯어보자"는 리뷰 진행 상황. 컨텍스트 컴팩트 이후에도 여기서부터 이어가면 됩니다.

## A. 데이터 수집·전처리 스크립트
- [x] 1. `image_crawler/kickboard_crawler/crawl_ddg.py` — DDGS 크롤러 (완료: 설정블록/핵심함수4개/main흐름/한계 설명함)
- [x] 2. `composite_violation.py` — rembg + 알파합성 위반/정상 이미지 생성 (완료: 설정블록/핵심함수4개/main흐름/seamlessClone→알파합성 전환/한계 설명함)
- [x] 3. `augment_board_lighting.py` — 도마 조명 증강 (완료: 목적/라벨재사용원리/조명변형3요소/train전용 안전장치 설명함)
- [x] 4. `resolve_occlusion_labels.py` — 가려짐 라벨 후처리 (완료: 원래 2단계 설계였으나 실제 라벨링 규칙이 "전체범위 추정 or 생략"으로 단순화되며 미사용 죽은코드가 된 이유 설명함)

## B. 라벨링 인프라
- [x] 5. `.venv_anylabeling` — X-AnyLabeling 격리 환경 (완료: include-system-site-packages=false 격리원리 + 실제 설치버전(PyQt6 6.6.1, anylabeling 0.4.36) 확인함)
- [x] 6. `labeling/classes.txt` — 8클래스 정의 (완료: ID순서/접두사-judge_image 연결/손칼 포함이유/분리설계 이유 설명함)
- [x] 7. 라벨링 규칙 (겹침/가려짐 처리 원칙) (완료: 겹침=독립인스턴스, 가려짐=추정or생략, 박스타이트니스 3원칙 정리함)

## C. 데이터셋 빌드
- [x] 8. JSON→YOLO txt 변환 로직 (완료: 저장된 파일 없어 재구성 설명 — 클래스매핑/좌표정규화/seed42 85:15분할 원리, 파일로 안 남은 이유 및 재현성 약점 짚음)
- [ ] 9. `labeling/dataset/data.yaml` + train/val 85:15 분할

## D. 학습·평가 노트북 (`yolomaskdetection_YOLO11_변환.ipynb`)
- [ ] 10. 데이터셋 업로드 섹션
- [ ] 11. 학습 셀 (imgsz=640, batch=-1, cache=True, seed=42, patience=50)
- [ ] 12. `judge_image()` 판정 규칙 함수
- [ ] 13. 발표용 시각화 셀 (맞은예시/틀린예시 자동 분류)
- [ ] 14. 영상 테스트 섹션

## E. 검증 결과
- [ ] 15. 정지 이미지 검증 결과
- [ ] 16. 영상 검증 결과 (AV1 코덱 이슈 포함)
- [ ] 17. 학습 곡선 해석 (과대적합 분석)

## F. 발표 자료
- [ ] 18. `claude_project_prompt.md`
- [ ] 19. `PPT_첨부자료/` 폴더 구성

**진행 중 이미 설명한 개념**: 해밍 거리(pHash 중복판정), 알파 합성 vs seamlessClone 차이 — 필요시 재설명 요청 가능.
