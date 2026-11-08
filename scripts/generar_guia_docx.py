"""Genera la guía Word desde Markdown con Open XML y librería estándar."""
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/GUIA_INTEGRAL_PROYECTO.md'
OUTPUT = ROOT / 'docs/Guia_integral_proyecto_redes_pluviales.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ET.register_namespace('w', W)


def tag(name):
    return f'{{{W}}}{name}'


def plain(text):
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return text.replace('**', '').replace('`', '')


def paragraph(parent, text, style='Normal'):
    p = ET.SubElement(parent, tag('p'))
    props = ET.SubElement(p, tag('pPr'))
    ET.SubElement(props, tag('pStyle'), {tag('val'): style})
    run = ET.SubElement(p, tag('r'))
    element = ET.SubElement(run, tag('t'), {'{http://www.w3.org/XML/1998/namespace}space': 'preserve'})
    element.text = plain(text)
    return p


def table(parent, rows):
    table_element = ET.SubElement(parent, tag('tbl'))
    props = ET.SubElement(table_element, tag('tblPr'))
    ET.SubElement(props, tag('tblStyle'), {tag('val'): 'TableGrid'})
    ET.SubElement(props, tag('tblW'), {tag('w'): '0', tag('type'): 'auto'})
    for row in rows:
        tr = ET.SubElement(table_element, tag('tr'))
        for value in row.strip('|').split('|'):
            cell = ET.SubElement(tr, tag('tc'))
            paragraph(cell, value.strip(), 'TableText')


def build_document(markdown):
    root = ET.Element(tag('document'))
    body = ET.SubElement(root, tag('body'))
    lines = markdown.splitlines(); i = 0; in_code = False
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            in_code = not in_code; i += 1; continue
        if in_code:
            paragraph(body, line or ' ', 'Code'); i += 1; continue
        if not line.strip(): i += 1; continue
        if line.startswith('|') and i+1 < len(lines) and re.match(r'^\|[ :|\-]+\|$', lines[i+1].strip()):
            rows = [line]; i += 2
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(lines[i]); i += 1
            table(body, rows); continue
        heading = re.match(r'^(#{1,3})\s+(.*)', line)
        if heading:
            level = len(heading.group(1))
            paragraph(body, heading.group(2), 'Title' if level == 1 else f'Heading{level-1}')
        elif line.startswith('- '): paragraph(body, '• ' + line[2:], 'Normal')
        elif re.match(r'^\d+\. ', line): paragraph(body, line, 'Normal')
        else: paragraph(body, line.removeprefix('> '), 'Normal')
        i += 1
    section = ET.SubElement(body, tag('sectPr'))
    ET.SubElement(section, tag('pgSz'), {tag('w'): '11906', tag('h'): '16838'})
    ET.SubElement(section, tag('pgMar'), {tag('top'): '1134', tag('right'): '1134', tag('bottom'): '1134', tag('left'): '1134'})
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:color w:val="17324C"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="270" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="173B60"/><w:sz w:val="42"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="0"/><w:spacing w:before="300" w:after="120"/></w:pPr><w:rPr><w:b/><w:color w:val="173B60"/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="1"/><w:spacing w:before="180" w:after="80"/></w:pPr><w:rPr><w:b/><w:color w:val="265675"/><w:sz w:val="25"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:rPr><w:sz w:val="19"/></w:rPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="B9CDDD"/><w:left w:val="single" w:sz="4" w:color="B9CDDD"/><w:bottom w:val="single" w:sz="4" w:color="B9CDDD"/><w:right w:val="single" w:sz="4" w:color="B9CDDD"/><w:insideH w:val="single" w:sz="4" w:color="B9CDDD"/><w:insideV w:val="single" w:sz="4" w:color="B9CDDD"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''
CONTENT = '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
DOC_RELS = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    with zipfile.ZipFile(OUTPUT, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CONTENT)
        z.writestr('_rels/.rels', ROOT_RELS)
        z.writestr('word/document.xml', build_document(SOURCE.read_text(encoding='utf8')))
        z.writestr('word/styles.xml', STYLES)
        z.writestr('word/_rels/document.xml.rels', DOC_RELS)
    print(OUTPUT)


if __name__ == '__main__':
    main()
