# -*- coding: utf-8 -*-
"""가려짐(_가려짐) 접미사 클래스를 원래 클래스로 합치는 후처리 스크립트

▶ 왜 필요한가?
  YOLO txt 라벨(class x y w h)에는 "가려짐 여부"를 저장할 칸이 없습니다.
  그래서 라벨링할 때 일부만 가려진 재료는 `재료_육류_가려짐`처럼
  "_가려짐" 접미사가 붙은 임시 클래스로 표시해두고, 학습 직전에 이 스크립트로
    1) 접미사를 떼어 원래 클래스(`재료_육류`)로 합치고
    2) 어떤 이미지의 몇 번째 박스가 가려져 있었는지 CSV로 남깁니다.
  (심하게 가려져서 아예 라벨링을 안 한 경우는 이 스크립트가 다룰 대상이 아닙니다 —
   그런 경우는 라벨링 단계에서 이미 박스 자체를 그리지 않아야 합니다.)

▶ 사용법
  1) [설정]의 LABELS_DIR, DATA_YAML을 본인 경로로 수정
  2) python resolve_occlusion_labels.py 실행
  3) LABELS_DIR의 txt 파일들이 "_가려짐 없는" 클래스 ID로 덮어써지고,
     같은 폴더에 occlusion_report.csv 가 생성됩니다.
"""
import os
import glob
import csv
import yaml

# ============================================================
#  [설정]
# ============================================================
LABELS_DIR = r"C:\choi\비전 프로젝트\dataset\labels"   # yolo txt 라벨들이 있는 폴더
DATA_YAML = r"C:\choi\비전 프로젝트\dataset\data.yaml"  # names 리스트가 들어있는 yaml
SUFFIX = "_가려짐"
REPORT_PATH = os.path.join(LABELS_DIR, "occlusion_report.csv")
# ============================================================


def load_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]


def build_id_maps(names):
    """접미사 클래스 id -> 원래 클래스 id 매핑, 원래 클래스 이름 -> id 매핑을 만든다"""
    name_to_id = {name: i for i, name in enumerate(names)}
    remap = {}  # occluded_id -> base_id
    for i, name in enumerate(names):
        if name.endswith(SUFFIX):
            base_name = name[: -len(SUFFIX)]
            if base_name not in name_to_id:
                raise ValueError(
                    f"'{name}'의 원래 클래스 '{base_name}'가 names 목록에 없습니다. "
                    f"클래스명을 확인하세요."
                )
            remap[i] = name_to_id[base_name]
    return remap


def main():
    names = load_names(DATA_YAML)
    remap = build_id_maps(names)
    if not remap:
        print("'_가려짐' 접미사가 붙은 클래스가 data.yaml에 없습니다. 할 일이 없습니다.")
        return

    print("가려짐 클래스 -> 원래 클래스 매핑:")
    for occ_id, base_id in remap.items():
        print(f"  [{occ_id}] {names[occ_id]}  ->  [{base_id}] {names[base_id]}")

    label_paths = glob.glob(os.path.join(LABELS_DIR, "*.txt"))
    report_rows = []
    changed_files = 0

    for path in label_paths:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        file_changed = False
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in remap:
                report_rows.append({
                    "file": os.path.basename(path),
                    "line": line_idx,
                    "occluded_class": names[cls_id],
                    "resolved_class": names[remap[cls_id]],
                })
                parts[0] = str(remap[cls_id])
                file_changed = True
            new_lines.append(" ".join(parts) + "\n")

        if file_changed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            changed_files += 1

    with open(REPORT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "line", "occluded_class", "resolved_class"])
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\n완료: {changed_files}개 파일 수정, 가려짐 표시 {len(report_rows)}건")
    print(f"기록 저장 -> {REPORT_PATH}")
    print("주의: data.yaml의 names에서 '_가려짐' 클래스들은 이제 지우고 nc를 base 클래스 수로 맞추세요.")


if __name__ == "__main__":
    main()
