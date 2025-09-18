import re, csv
from pathlib import Path

readme_path = Path("/workspace/admmed/README.md")
out_csv = Path("/workspace/admmed/papers.csv")
text = readme_path.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

entries = []
current_category = None

bold_title_re = re.compile(r"^\*\*(.+?)\*\*.*$")
ital_auth_re = re.compile(r"^\*(.+?)\*.*$")
date_venue_re = re.compile(r"^\[(.+?)\]\s*\[(.+?)\].*$")
category_re = re.compile(r"^###\s+(.+?)\s*$")
subcategory_re = re.compile(r"^####\s+(.+?)\s*$")
links_pattern = re.compile(r"\[\[(?P<label>[^\]]+)\]\((?P<url>[^\)]+)\)\]", re.IGNORECASE)


def infer_model(name: str) -> str:
    n = name.lower()
    model = []
    if any(k in n for k in ["ddpm", "denoising diffusion", "diffusion probabilistic"]):
        model.append("DDPM类")
    if "score" in n or "score-based" in n:
        model.append("Score-based")
    if "latent" in n:
        model.append("潜空间/Latent Diffusion")
    if "conditional" in n or "conditioned" in n:
        model.append("条件扩散")
    if "schrodinger" in n or "schrödinger" in n:
        model.append("Schrödinger Bridge")
    if "ode" in n and "diffusion" in n:
        model.append("扩散ODE")
    if "sde" in n:
        model.append("SDE")
    if "flow" in n:
        model.append("流模型/Flow")
    if "gan" in n:
        model.append("GAN")
    if "masked" in n or "mask" in n:
        model.append("掩码/Masked机制")
    if "bayesian" in n:
        model.append("贝叶斯/不确定性")
    if "normative" in n:
        model.append("规范/Normative")
    if not model:
        model.append("通用/未明确")
    seen = []
    for m in model:
        if m not in seen:
            seen.append(m)
    return ", ".join(seen)


def infer_method(name: str) -> str:
    n = name.lower()
    if "counterfactual" in n or "pseudo-healthy" in n:
        return "伪健康/反事实重建以显著化异常"
    if "reconstruction" in n or "restoration" in n:
        return "重建/复原驱动"
    if "anomaly" in n:
        return "异常检测/定位"
    if "segmentation" in n:
        return "分割"
    if "registration" in n:
        return "配准"
    if "classification" in n:
        return "分类"
    if "super-resolution" in n or "super resolution" in n:
        return "超分"
    if "denois" in n:
        return "去噪"
    if "translation" in n:
        return "图像到图像翻译"
    if "text-to-image" in n or "text to image" in n:
        return "文本生成图像"
    if "editing" in n or "edit" in n:
        return "编辑"
    return "未归纳"


def map_task(cat: str) -> str:
    c = (cat or "").strip().lower()
    mapping = {
        "anomaly detection": "异常检测/定位",
        "denoising": "去噪",
        "segmentation": "分割",
        "image-to-image translation": "图像到图像翻译",
        "reconstruction": "重建",
        "image generation": "生成",
        "text-to-image": "文生图",
        "registration": "配准",
        "classification": "分类",
        "object detection": "目标检测",
        "image restoration": "图像复原",
        "inpainting": "图像修复/补全",
        "super resolution": "超分",
        "enhancement": "增强",
        "editing": "编辑",
        "adversarial attacks": "对抗攻击/鲁棒性",
        "fairness": "公平性",
        "time series": "时间序列",
        "audio": "音频",
        "multi-task": "多任务",
        "other applications": "其他应用",
    }
    return mapping.get(c, cat)


in_papers_section = False
for idx, line in enumerate(lines):
    if line.strip() == "## Papers":
        in_papers_section = True
        continue
    if in_papers_section:
        mcat = category_re.match(line)
        msub = subcategory_re.match(line)
        if mcat:
            current_category = mcat.group(1).strip()
            continue
        if msub and (current_category and current_category.lower() in ("image restoration",)):
            current_category = msub.group(1).strip()
            continue
        mtitle = bold_title_re.match(line)
        if mtitle:
            title = mtitle.group(1).strip()
            authors = ""
            venue = ""
            date = ""
            links = []
            # Look ahead a few lines for details
            j = idx + 1
            for k in range(j, min(j + 8, len(lines))):
                s = lines[k].strip()
                if not s:
                    break
                if bold_title_re.match(lines[k]) or category_re.match(lines[k]) or subcategory_re.match(lines[k]):
                    break
                ma = ital_auth_re.match(lines[k])
                if ma and not authors:
                    authors = ma.group(1).strip()
                    continue
                mdv = date_venue_re.match(s)
                if mdv and not date and not venue:
                    date = mdv.group(1).strip()
                    venue = mdv.group(2).strip()
                    continue
                for lm in links_pattern.finditer(lines[k]):
                    label = lm.group("label")
                    url = lm.group("url")
                    links.append(f"{label}:{url}")
            task = map_task(current_category)
            model = infer_model(title)
            method = infer_method(title)
            entries.append({
                "任务类别": task,
                "原始类别": current_category,
                "论文题目": title,
                "作者": authors,
                "日期": date,
                "发表/场所": venue,
                "模型架构(推断)": model,
                "方法要点(推断)": method,
                "链接": " | ".join(links),
                "效果摘要": "（README未提供，详见论文）",
            })

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    if entries:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
        writer.writerows(entries)
print(f"Parsed entries: {len(entries)} -> {out_csv}")
