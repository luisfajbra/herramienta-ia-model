"""Produce una copia de la presentación con la conclusión de auditoría corregida."""
from pathlib import Path
import hashlib
import zipfile

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Presentacion_MLSMFP_08-11_entrenamiento_conjunto_y_reserva.pptx'
OUTPUT = ROOT / 'Presentacion_MLSMFP_08-11_entrenamiento_conjunto_revision_auditoria.pptx'

def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prs = Presentation(SOURCE)
    slide = prs.slides[55]
    changes = {
        'MLP logró menor RMSE de volumen en esta reserva.': 'MLP tiene menor RMSE en la reserva completa.',
        'Un hidrograma sostenido concentra gran parte del error cuadrático.': 'Sin un hidrograma sostenido, XGBoost tiene menor RMSE (análisis posterior).',
    }
    count = 0
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                for old, new in changes.items():
                    if old in run.text:
                        run.text = run.text.replace(old, new)
                        count += 1
    if count != len(changes) or len(prs.slides) != 60:
        raise ValueError('La presentación de entrada no coincide con la versión esperada')
    prs.save(OUTPUT)
    with zipfile.ZipFile(OUTPUT) as package:
        if package.testzip() is not None or len(package.namelist()) != len(set(package.namelist())):
            raise ValueError('El paquete PPTX no pasó la comprobación')
    print('Presentación:', OUTPUT)
    print('SHA-256:', hashlib.sha256(OUTPUT.read_bytes()).hexdigest())

if __name__ == '__main__':
    main()
