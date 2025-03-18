import os
import re
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from mecab import MeCab

app = Flask(__name__)

# .env 파일 로드
BASE_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
load_dotenv(os.path.join(BASE_DIR, "openai.env"))

# API 키 설정
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)
token_limit = 512

# 비속어 리스트 로드
with open(
    os.path.join(BASE_DIR, "korean_profanity_v1.0.1.json"), encoding="utf-8"
) as f:
    profanity_list = json.load(f)

# MeCab 형태소 분석기 초기화
tagger = MeCab()


@app.route("/")
def home():
    return "Team SAURUS Flask Server is running..."


@app.route("/generate_letter", methods=["POST"])
def generate_letter():
    try:
        # 안드로이드에서 전송한 JSON 파싱
        data = request.json

        # author, topic, style 키가 들어있다고 가정
        author = data.get("author", "무명 작가")
        documentType = data.get("documentType", "단문")
        scenario = data.get("scenario", "주제 없음")
        app.logger.info(
            "\n--- Request ---\n"
            f"author: {author}\n"
            f"documentType: {documentType}\n"
            f"scenario: {scenario}\n"
        )

        # scenario 값이 한글 자음, 모음만으로 이루어진 경우 검사 (공백 포함)
        if not re.search(r"[A-Za-z가-힣]", scenario):
            app.logger.warning("Invalid scenario value.")
            return jsonify(
                {
                    "isSuccessful": False,
                    "result": "당신의 이야기를 조금만 더 들려주실 수 있나요?",
                }
            )

        # 1) Mecab으로 토큰화
        scenario_tokens = tagger.morphs(scenario)

        # 2) 비속어 검사 (원본 문장 + 토큰 리스트)
        found_words = [
            word
            for word in profanity_list
            if (word in scenario) or (word in scenario_tokens)
        ]

        # 중복 제거
        found_words = list(set(found_words))

        if found_words:
            app.logger.warning(f"\nFound forbidden words: {found_words}\n")
            response_words = ", ".join(found_words)
            return jsonify(
                {
                    "isSuccessful": False,
                    "result": f"민감한 표현의 단어가 포함되었습니다. : {response_words}",
                }
            )

        # 시스템 메시지
        system_prompt = (
            "당신은 사용자가 요청한 특정 작가의 문체와 감성을 정밀하게 재현할 수 있는 글쓰기 전문 AI 보조입니다. "
            f"{token_limit}토큰을 초과하지 않는 범위에서 하나의 단락으로 구성된 글을 작성하되 "
            "글이 중간에 끊기지 않고 완결된 문장으로 마무리되도록 해 주세요. "
            "글이 너무 길어질 것 같다면 요약하되, 마지막 문장은 반드시 문장 부호를 포함하여 완전히 끝맺어야 합니다. "
            "작가의 대표적인 작품 분위기, 문체적 특징 및 표현 방식을 명확히 반영해 주세요. "
            "단, 텍스트 이외의 생성형 이미지나 사진, 그림 등 시각적 콘텐츠는 생성하거나 요청해서는 안 됩니다."
        )

        # 유저 메시지
        user_prompt = (
            "다음 조건에 따라 글을 작성해 주세요.\n\n"
            f"- 작가: {author}\n"
            f"- 글의 종류: {documentType}\n"
            f"- 글의 주제 및 상황: {scenario}\n\n"
            "조건의 설명을 포함하지 말고, 글만 생성해야 합니다."
            "글이 중간에 끊기지 않고, 마지막 문장까지 매끄럽게 이어져야 합니다."
        )

        # --- 스트리밍 방식으로 ChatCompletion 호출 ---
        chat = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=token_limit,
            stream=True,  # 스트리밍 모드
        )

        generated_text = ""
        for chunk in chat:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                generated_text += delta.content

        '''
        # 마지막 문장이 완결되었는지 확인 ('.', '!', '?' 등으로 끝나야 함)
        generated_text = generated_text.strip()
        if not generated_text.endswith((".", "!", "?")):
            # 마지막 문장부호가 나타난 마지막 인덱스를 찾음
            last_punct_idx = max(generated_text.rfind(p) for p in ".!?")
            if last_punct_idx != -1:
                # 마지막 완결된 문장까지만 남김
                generated_text = generated_text[: last_punct_idx + 1]
                app.logger.info("[Deleted] Incomplete sentence removed.\n")
            else:
                app.logger.info(
                    "No sentence-ending punctuation found; returning as is.\n"
                )
        '''
        return jsonify(
            {
                "isSuccessful": True,
                "result": generated_text,
            }
        )

    except Exception as e:
        app.logger.exception("Error in generate_letter")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6947, debug=True)
