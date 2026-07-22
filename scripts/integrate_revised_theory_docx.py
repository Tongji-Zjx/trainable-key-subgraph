"""Integrate the revised Phi/Lemma/Theorem blocks into the main DOCX body."""

from __future__ import absolute_import, print_function

import copy
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)


def _tag(local):
    return "{{{}}}{}".format(W, local)


def _paragraph_text(paragraph):
    return "".join(node.text or "" for node in paragraph.iter() if node.tag == _tag("t"))


def _body_paragraphs(body):
    return [node for node in list(body) if node.tag == _tag("p")]


def _find(paragraphs, prefix):
    matches = [p for p in paragraphs if _paragraph_text(p).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("expected one paragraph beginning with {!r}, found {}".format(prefix, len(matches)))
    return matches[0]


def _find_first(paragraphs, prefix):
    for paragraph in paragraphs:
        if _paragraph_text(paragraph).startswith(prefix):
            return paragraph
    raise ValueError("paragraph beginning with {!r} was not found".format(prefix))


def _replace_text(paragraph, text):
    paragraph = copy.deepcopy(paragraph)
    ppr = paragraph.find(_tag("pPr"))
    first_rpr = None
    for run in paragraph.findall(_tag("r")):
        rpr = run.find(_tag("rPr"))
        if rpr is not None:
            first_rpr = copy.deepcopy(rpr)
            break
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = ET.SubElement(paragraph, _tag("r"))
    if first_rpr is not None:
        run.append(first_rpr)
    text_node = ET.SubElement(run, _tag("t"))
    text_node.set("{{{}}}space".format(XML), "preserve")
    text_node.text = text
    return paragraph


def _replace_range(body, first, last_exclusive, replacements):
    children = list(body)
    start = children.index(first)
    end = children.index(last_exclusive)
    for child in children[start:end]:
        body.remove(child)
    insert_at = start
    for replacement in replacements:
        body.insert(insert_at, copy.deepcopy(replacement))
        insert_at += 1


def integrate(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    with zipfile.ZipFile(str(source), "r") as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    body = root.find(".//{}".format(_tag("body")))
    paragraphs = _body_paragraphs(body)

    revised_phi_heading = _find(paragraphs, "一、结构统计映射 Φ 的具体定义")
    revised_lemma1_heading = _find(paragraphs, "二、修改后的 Lemma 1")
    revised_lemma2_heading = _find(paragraphs, "三、修改后的 Lemma 2")
    revised_theorem_heading = _find(paragraphs, "四、修改后的 Theorem 1")
    revised_corollary_heading = _find(paragraphs, "五、推论")

    children = list(body)
    phi_replacements = children[
        children.index(revised_phi_heading) + 1 : children.index(revised_lemma1_heading)
    ]
    lemma1_replacements = children[
        children.index(revised_lemma1_heading) : children.index(revised_lemma2_heading)
    ]
    lemma2_replacements = children[
        children.index(revised_lemma2_heading) : children.index(revised_theorem_heading)
    ]
    theorem_replacements = children[
        children.index(revised_theorem_heading) : children.index(revised_corollary_heading)
    ]

    # Remove the supplement's meta-numbering after moving the revised material
    # into the corresponding main-theory sections.
    lemma1_replacements[0] = _replace_text(
        lemma1_replacements[0], "Lemma 1：结构统计变化差异引理"
    )
    lemma2_replacements[0] = _replace_text(
        lemma2_replacements[0], "Lemma 2：关键子图判别结构变化保持"
    )
    theorem_replacements[0] = _replace_text(
        theorem_replacements[0], "Theorem 1：关键子图演化判别能力定理"
    )

    paragraphs = _body_paragraphs(body)
    old_definition = _find(paragraphs, "定义5：")
    theory_heading = _find(paragraphs, "七、理论引理")
    _replace_range(body, old_definition, theory_heading, phi_replacements)

    paragraphs = _body_paragraphs(body)
    old_lemma1 = _find_first(paragraphs, "Lemma 1：动态图结构差异导致演化差异")
    old_lemma2 = _find_first(paragraphs, "Lemma 2：关键子图变化保持原图变化")
    _replace_range(body, old_lemma1, old_lemma2, lemma1_replacements)

    paragraphs = _body_paragraphs(body)
    old_lemma2 = _find_first(paragraphs, "Lemma 2：关键子图变化保持原图变化")
    lemma3 = _find_first(paragraphs, "Lemma 3：演化编码保持结构差异")
    _replace_range(body, old_lemma2, lemma3, lemma2_replacements)

    paragraphs = _body_paragraphs(body)
    old_theorem = _find_first(paragraphs, "Theorem 1：关键子图演化表达能力定理")
    corollaries_heading = _find(paragraphs, "九、推论")
    _replace_range(body, old_theorem, corollaries_heading, theorem_replacements)

    paragraphs = _body_paragraphs(body)
    intro = paragraphs[2]
    intro_new = _replace_text(
        intro,
        "针对审稿意见，本文不再由类别条件动态图分布不同直接推出结构变化统计量必然不同，而是将具有明确语义的结构统计映射 Φ 具体化，并在存在类条件结构变化矩差异的前提下，论证关键子图对判别结构变化的保持能力。",
    )
    body.insert(list(body).index(intro), intro_new)
    body.remove(intro)

    paragraphs = _body_paragraphs(body)
    summary = _find(paragraphs, "修订后的理论核心：")
    summary_new = _replace_text(
        summary,
        "修订后的理论核心：可解释结构统计变化存在类别差异 → 全图结构变化分布不同 → 关键子图在判别结构统计子空间中以有界误差保持这种变化 → 当保持误差足够小时，关键子图演化变化仍包含类别相关的动态图结构信息。该版本避免把任意抽象映射或单纯的动态图分布差异当作充分依据。",
    )
    body.insert(list(body).index(summary), summary_new)
    body.remove(summary)

    paragraphs = _body_paragraphs(body)
    appendix = _find(paragraphs, "附录：引理与定理的详细数学证明补充")
    sect_pr = body.find(_tag("sectPr"))
    if sect_pr is None:
        raise ValueError("document has no section properties")
    _replace_range(body, appendix, sect_pr, [])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent)) as temporary:
        staged = Path(temporary) / output.name
        with zipfile.ZipFile(str(source), "r") as source_zip, zipfile.ZipFile(
            str(staged), "w", compression=zipfile.ZIP_DEFLATED
        ) as target_zip:
            for item in source_zip.infolist():
                data = (
                    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
                    ET.tostring(root, encoding="utf-8")
                    if item.filename == "word/document.xml"
                    else source_zip.read(item.filename)
                )
                target_zip.writestr(item, data)
        shutil.copy2(str(staged), str(output))
    return output


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    integrate(
        here / "docs/Key_Subgraph_Evolution_Theory_Revised_Theorem_and_Lemma.docx",
        here / "docs/Key_Subgraph_Evolution_Theory_Integrated_Final.docx",
    )
