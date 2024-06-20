from typing import Dict, Union
from pathlib import Path

from os.path import dirname, abspath, join
from yaml import safe_load
from uuid import uuid4
from shutil import copy

from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

from markdown_pdf import MarkdownPdf, Section

PATH_TO_DIR = dirname(abspath(__file__))
CONTEXT_LEN = 15000
MAX_PRED_LEN = 512
TEMPERATURE = 0.2


def get_transcript(path: Union[str, Path]) -> str:
    """Загружает расшифровку переговоров из файла

    Args:
        path (Union[str, Path]): путь к файлу с транскрипцией переговоров

    Returns:
        str: текст переговоров
    """
    with open(path, "r") as f:
        transcript = f.read()

    return transcript


def get_questions(path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """Загружает список вопросов из yaml

    Args:
        path (Union[str, Path]): путь к yaml файлу

    Returns:
        Dict[str, Dict[str, str]]: словарь с вопросами и соответствующими структурными шаблонами ответов
    """
    with open(path, "r") as f:
        questions = safe_load(f)

    return questions


def get_answer(transcript: str, questions: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Генерирует ответы на вопросы questions по контексту transcript

    Args:
        transcript (str): расшифровка переговоров
        questions (Dict[str, Dict[str, str]]): вопросы к расшифровке

    Returns:
        Dict[str, str]: словарь с ответами модели
    """
    model = Ollama(
        model="qwen2:7b",
        num_ctx=CONTEXT_LEN,
        num_predict=MAX_PRED_LEN,
        temperature=TEMPERATURE,
    )

    template = """Ты опытный бизнесс-ассистент и специалист по поиску ключевой информации в тексте. 
    Твоя задача изучить расшифровку переговоров и ответить на заданный вопрос по этой расшифровке.
    Формулировки в ответе должны быть короткие и четкие. Ответ должен соответствовать шаблону ответа.
    Если в тексте нет требуемой информации, напиши, что информация отсутствует.
    Расшифровка переговоров: {transcript}
    Вопрос: {question}
    Шаблон ответа: {ans_template}"""

    promt = PromptTemplate(
        input_variables=["transcript", "question", "ans_template"], template=template
    )
    chain = promt | model
    result = {}

    for key, value in questions.items():
        field = key
        question = value["question"]
        ans_template = value["ans_template"]

        print(f"Вопрос в обработке: {question}")
        result[field] = chain.invoke(
            input={
                "transcript": transcript,
                "question": question,
                "ans_template": ans_template,
            }
        )

    return result


def get_summary_file(
    path: Union[str, Path], result: Dict[str, str], make_pdf: bool = True
) -> None:
    """Генерирует файл с результатами суммаризации по соответствующему шаблону

    Args:
        path (Union[str, Path]): путь к шаблону файла
        result (Dict[str, str]): ответы модели
        make_pdf (Optional[bool], optional): Если True генерирует pdf файл. Defaults to True
    """
    file_name = f"summary{str(uuid4())}.md"
    path_to_file = join(PATH_TO_DIR, file_name)
    path_to_logo = join("template", "img.svg")
    copy(path, path_to_file)

    with open(path_to_file, "r+") as f:
        lines = f.readlines()
        lines[1] = lines[1].replace(":::path_to_img:::", f"{path_to_logo}")
        
        for field, ans in result.items():
            lines[lines.index(f":::{field}:::\n")] = ans

        text = ''.join(lines)
        f.seek(0)
        f.write(text)

        if make_pdf:
            path_to_pdf = path_to_file[:-2] + "pdf"
            pdf = MarkdownPdf()
            pdf.add_section(Section(text, toc=False, root=PATH_TO_DIR, borders=(10, 10, -10, -10)))
            pdf.save(path_to_pdf)


def main() -> None:
    """Основная функция
    """
    path_to_transcript = join(PATH_TO_DIR, "transcript.md")
    path_to_template = join(PATH_TO_DIR, "template", "template.md")
    path_to_questions = join(PATH_TO_DIR, "template", "questions.yaml")

    transcript = get_transcript(path_to_transcript)
    questions = get_questions(path_to_questions)
    result = get_answer(transcript, questions)
    print("Генерация файла")
    get_summary_file(path_to_template, result)


if __name__ == "__main__":
    main()
