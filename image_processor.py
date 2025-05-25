# -процесс обработки изображений
import easyocr
import cv2
import numpy as np
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from loguru import logger
from PIL import Image
import pytesseract
import re
import os
from difflib import get_close_matches
from config import TEMP_DIR 
import hashlib 
import hunspell

class ImageProcessor:
    
    def __init__(self):
        with open('ru.txt', 'r', encoding='utf-8') as f:
            self.custom_vocab = [line.strip().lower() for line in f]

        self.reader = easyocr.Reader(
            lang_list=['ru', 'en'],
            detector=True,
            detect_network='dbnet18',
            model_storage_directory="model",
            download_enabled=True,
            quantize=False,  
            )
        
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        self.temp_dir = TEMP_DIR
        self.allowlist = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_,.?!:;/()@_&%+–—"\''
        self.textCorrector = self.TextCorrector('./pn/ru_RU.dic', './pn/ru_RU.aff', './pn/en_US.dic', './pn/en_US.aff')

    def preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}")
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 7, 21)

            img = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)

            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                27, 15
            )

            _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            denoised = cv2.fastNlMeansDenoising(thresh)
            return denoised

        except Exception as e:
            logger.error(f"Ошибка при предобработке: {e}")
            return None

        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            return None

    def extract_text_tesseract(self, image_path: str) -> str:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="rus+eng", config='--psm 6 --oem 1')
        return text
    
    def extract_text(self, image_path: str) -> str:
        try:
            processed_img = self.preprocess_image(image_path)
            if processed_img is None:
                return ""

            results = self.reader.readtext(
                processed_img,
                allowlist=self.allowlist,
                detail=0,
                paragraph=True
            )

            text = " ".join(results)
            return text
        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return ""

    def process_text(self, text: str) -> str:
        try:
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.parse_syntax(self.syntax_parser)
            doc.tag_ner(self.ner_tagger)

            for span in doc.spans:
                span.normalize(self.morph_vocab)

            return " ".join([
                token.text for token in doc.tokens
                if not token.pos in {'PUNCT', 'SYMB'}
            ])
        except Exception as e:
            logger.error(f"Ошибка при обработке текста: {e}")
            return text

    def clean_ocr_text(self, text: str) -> str:
        text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)
        text = re.sub(r'[^\w\s.,!?;:@/()\-–—]', '', text)
        mistakes = {
            r'\bсгг\b': 'сеть',
            r'\b[З3]адание\b': 'задание',
            r'\bв0\b': 'во'
        }

        text = re.sub(r'\.\s*\.\s*\.', '...', text) 
        text = re.sub(r'[•·∙·•]', '', text)
        text = re.sub(r'\s*\.\s*', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        for pattern, fix in mistakes.items():
            text = re.sub(pattern, fix, text, flags=re.IGNORECASE)
        return text.strip()

    def calculate_image_hash(self, image_path: str) -> str:
        """Вычисляет хеш изображения для определения дубликатов"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}")
                
            img = cv2.resize(img, (8, 8))
            avg = img.mean()
            hash_bits = img > avg
            hash_str = ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
            hash_hex = hex(int(hash_str, 2))[2:].zfill(16)
            
            return hash_hex
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении хеша изображения {image_path}: {e}")
            return None
            
    def is_duplicate(self, image_path: str, image_hash: str) -> bool:
        """Проверяет, является ли изображение дубликатом"""
        try:
            current_hash = self.calculate_image_hash(image_path)
            if not current_hash:
                return False
            return current_hash == image_hash
            
        except Exception as e:
            logger.error(f"Ошибка при проверке дубликата {image_path}: {e}")
            return False

    def process_image(self, image_path: str) -> str:
        """Обрабатывает изображение и возвращает распознанный текст"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Файл не найден: {image_path}")
            file_size = os.path.getsize(image_path)
            image_hash = self.calculate_image_hash(image_path)
            if not image_hash:
                raise ValueError("Не удалось вычислить хеш изображения")
                
            raw_text = self.extract_text(image_path)
            raw_text_tesseract = self.extract_text_tesseract(image_path)
            raw_text = self.clean_ocr_text(raw_text)
            raw_text_tesseract = self.clean_ocr_text(raw_text_tesseract)

            corrected_text = self.textCorrector.correct_vocub(raw_text)
            corrected_text_tesseract = self.textCorrector.correct_vocub(raw_text_tesseract)
            processed_text = self.process_text(corrected_text)
            processed_text_tesseract = self.process_text(corrected_text_tesseract)

            return processed_text
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {image_path}: {e}")
            return None

    def cleanup(self):
        """Очищает временные файлы"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
        except Exception as e:
            logger.error(f"Ошибка при очистке временных файлов: {e}") 
    
    # внутренний класс
    class TextCorrector: 
        def __init__(self, dic_path_ru, aff_path_ru, dic_path_en, aff_path_en):
            self.hunspell_ru = hunspell.HunSpell(dic_path_ru, aff_path_ru)
            self.hunspell_en = hunspell.HunSpell(dic_path_en, aff_path_en)

        def correct_vocub(self, text: str) -> str:
            words = text.split()
            corrected = []

            for word in words:
                if self.is_russian(word):
                    if self.hunspell_ru.spell(word):
                        corrected.append(word)
                    else:
                        suggestions = self.hunspell_ru.suggest(word)
                        corrected.append(suggestions[0] if suggestions else word)
                else:
                    if self.hunspell_en.spell(word):
                        corrected.append(word)
                    else:
                        suggestions = self.hunspell_en.suggest(word)
                        corrected.append(suggestions[0] if suggestions else word)

            return ' '.join(corrected)


        def is_russian(self, word: str) -> bool:
            return any('а' <= char <= 'я' or 'А' <= char <= 'Я' for char in word)



