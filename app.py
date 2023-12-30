import os
import gradio as gr
import time
import json
import base64
import threading
import PIL.Image
import google.generativeai as genai
from openai import (
    OpenAI, AuthenticationError, NotFoundError, BadRequestError
)

# GPT用設定
GPT_MODEL_DF = "gpt-4-1106-preview"
# GPT_MODEL_DF = "gpt-3.5-turbo-1106"
GPT_MODEL_V = "gpt-4-vision-preview"
SYS_PROMPT_JA = "あなたは優秀なアシスタントです。回答は日本語でお願いします。"
SYS_PROMPT_EN = "You are a helpful assistant. Please answer in English."

# 設定値
MAX_CHAR = 1000
MAX_TOKENS = 1000
DETAIL = "low"

# コードなど
file_format = {".png", ".jpeg", ".jpg", ".webp", ".gif", ".PNG", ".JPEG", ".JPG", ".WEBP", ".GIF"}
lang_code = {'Japanese': "ja", 'English': "en"}
gpt_model_list = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]


# 各関数定義
def set_state(openai_key, gpt_model, lang, state):
    """ 設定タブの情報をセッションに保存する関数 """

    state["openai_key"] = openai_key
    state["gpt_model"] = gpt_model
    state["lang"] = lang_code[lang]

    return state


def init(state, text, image):
    """ 入力チェックを行う関数 """

    err_msg = ""

    if not text:

        # テキスト未入力
        err_msg = "テキストを入力して下さい。"

        return state, err_msg

    elif image:

        # 入力画像のファイル形式チェック
        root, ext = os.path.splitext(image)

        if ext not in file_format:

            # ファイル形式チェック
            err_msg = "指定した形式のファイルをアップしてください。（注意事項タブに記載）"

            return state, err_msg

    if state["client"] is None:   # 初回起動時

        # クライアント新規作成
        client = OpenAI()

        # Gemini用設定
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        # 各model生成
        model = genai.GenerativeModel('gemini-pro')
        model_vision = genai.GenerativeModel('gemini-pro-vision')

        # Chat新規作成
        chat_session = model.start_chat()

        # セッションにセット
        state["client"] = client
        state["model_vision"] = model_vision
        state["chat_ses"] = chat_session

        return state, err_msg


def raise_exception(err_msg):
    """ エラーの場合例外を起こす関数 """

    if err_msg != "":
        raise Exception("これは入力チェックでの例外です。")

    return


def user(user_message, chat_1, chat_2):
    return "", chat_1 + [[user_message, None]], chat_2 + [[user_message, None]]


def get_response(state, image, chat_1, chat_2):

    # 戻り値セット用変数
    return_gpt = [None] * 2
    return_gemini = [None] * 2
    err_msg = ""

    # セッションからセット
    # sys_prompt = state["system_prompt"]
    lang = state["lang"]
    client = state["client"]
    model_vision = state["model_vision"]
    chat_ses = state["chat_ses"]

    # 入力されたテキストセット
    user_msg2 = chat_2[-1][0]

    # それぞれの回答を得るスレッドを作る
    thread_gpt = threading.Thread(target=request_gpt, args=(client, lang, image, chat_1, return_gpt))
    thread_gemi = threading.Thread(target=request_gemini, args=(model_vision, chat_ses, image, user_msg2, return_gemini))

    # スレッドの処理を開始
    thread_gpt.start()
    thread_gemi.start()

    # スレッドの処理終了を待つ
    thread_gpt.join()
    thread_gemi.join()

    # エラーメッセージ設定
    if return_gpt[1] :

        err_msg = return_gpt[1]

    if return_gemini[1] :

        err_msg = return_gemini[1]

    # chat_1,chat_2は変更しないがチャット画面に（…）を表示するた加える
    return return_gpt[0], return_gemini[0], None, err_msg, chat_1, chat_2


def bot(gpt_msg, gemini_msg, chat_1, chat_2):

    chat_1[-1][1] = ""
    chat_2[-1][1] = ""

    # 文字列を分割
    char_list1 = list(gpt_msg)
    char_list2 = list(gemini_msg)

    for i in range(MAX_CHAR):

        if i <= len(char_list1)-1:

            chat_1[-1][1] += char_list1[i]

        if i <= len(char_list2)-1:

            chat_2[-1][1] += char_list2[i]

        # 0.03秒おきに1文字ずつ表示
        time.sleep(0.03)

        yield chat_1, chat_2

        if i >= len(char_list1)-1 and i >= len(char_list2)-1:

            break


def request_gemini(model_vision, chat_ses, image, text, return_arg):

    if image is None:

        try:

            # テキストのみの場合
            responses = chat_ses.send_message(text)

            return_arg[0] = responses.text

        except Exception as e:
            print(e)
            return_arg[0] = "[Geminiでエラーが発生しました。お手数ですがやり直して下さい。]"
            return_arg[1] = str(e)

    else:

        try:

            # 画像を開く
            pil_img = PIL.Image.open(image)

            # gemini_visionで回答を受け取る
            responses = model_vision.generate_content([text, pil_img])
            responses.resolve()

            return_arg[0] = responses.text

        except Exception as e:
            print(e)
            return_arg[0] = "[Gemini_Visionでエラーが発生しました。お手数ですがやり直して下さい。]"
            return_arg[1] = str(e)


def encode_image(image_path):
    """ base64エンコード用関数 """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def request_gpt(client, lang, image, chat, return_arg):

    # return_arg[0] = "テスト"

    # return

    if image is None:

        model=GPT_MODEL_DF

        messages = make_messages(lang, chat)

    else:

        model=GPT_MODEL_V

        # 画像をbase64に変換
        base64_image = encode_image(image)

        prompt = chat[-1][0]

        # メッセージの作成
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": DETAIL,
                        }
                    },
                ],
            }
        ]

    try:

        # gpt-4-visionに問い合わせて回答を表示
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
        )

        return_arg[0] = response.choices[0].message.content

    except BadRequestError as e:
        print(e)
        return_arg[0] = "[OpenAIリクエストエラーです。画像などポリシー違反でないか確認して下さい。]"
        return_arg[1] = str(e)
    except Exception as e:
        print(e)
        return_arg[0] = "[GPTへのリクエストでエラーが発生しました。お手数ですがやり直して下さい。]"
        return_arg[1] = str(e)


def make_messages(lang, history):
    """ "history"をGPT用の履歴"messages"に変換する関数 """

    # システムプロンプト設定
    if lang == "ja":
        messages = [{"role": "system", "content": SYS_PROMPT_JA}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT_EN}]

    for msg in history:

        messages.append({"role": "user", "content": msg[0]})

        if msg[1] is not None:    # 最後は空のためセットしない

            messages.append({"role": "assistant", "content": msg[1]})

    return messages


def clear_click(state):
    """ クリアボタンクリック時 """

    # セッションの一部をリセット
    state["client"] = None
    state["model_vision"] = None
    state["chat_ses"] = None
    return state


with gr.Blocks() as demo:

    title = "GPT-4 VS Gemini Pro"
    message = "・質問に対しGPT-4とGemini Proが同時に回答します。<br>"
    message += "・画像は保存されませんので、再度問いかけを行う際はもう1度画像入力してください。<br>"
    message += "・（「さっきの画像で…」のような問いかけはできません。）<br>"
    # message += "・<br>"
    # message += "・テスト中でAPIKEY無しで動きます。<br>"
    message += "・Geminiの動画での紹介はこちら→https://www.youtube.com/watch?v=pq-SRTCF5M0<br>"

    gr.Markdown("<h2>" + title + "</h2><h3>" + message + "</h3>")

    state = gr.State({
        # "system_prompt": SYS_PROMPT_DEFAULT,
        "openai_key" : "",
        "google_key" : "",
        "gpt_model" : GPT_MODEL_DF,
        "lang" : "ja",
        "client" : None,
        "model_vision" : None,
        "chat_ses" : None,
    })

    with gr.Tab("GPT-4 VS Gemini") as maintab:

      # 各コンポーネント定義
      with gr.Row():
        chat_1 = gr.Chatbot(label="GPT")
        chat_2 = gr.Chatbot(label="Gemini")

      text_msg = gr.Textbox(label="テキスト")

      # 返答セット用（非表示）
      gpt_msg = gr.Textbox(visible=False)
      gemini_msg = gr.Textbox(visible=False)

      with gr.Row():
          btn = gr.Button("送信")
          btn_clear = gr.ClearButton(value="リセット", components=[chat_1, chat_2, text_msg])

      image = gr.Image(label="ファイルアップロード", type="filepath",interactive = True)
      sys_msg = gr.Textbox(label="システムメッセージ", interactive = False)

    with gr.Tab("設定") as set:
      openai_key = gr.Textbox(label="OpenAI API Key", visible=False)
      gpt_model = gr.Dropdown(choices=gpt_model_list, value = GPT_MODEL_DF, label="GPT Model", interactive = True)
      lang = gr.Dropdown(choices=["Japanese", "English"], value = "Japanese", label="Language", interactive = True)

    # 設定変更時
    maintab.select(set_state, [openai_key, gpt_model, lang, state], state)

    # 問い合わせ実行
    text_msg.submit(init, [state, text_msg, image], [state, sys_msg], queue=False).then(
        raise_exception, sys_msg, None).success(
        user, [text_msg, chat_1, chat_2], [text_msg, chat_1, chat_2], queue=False).success(
        get_response, [state, image, chat_1, chat_2],  [gpt_msg, gemini_msg, image, sys_msg, chat_1, chat_2]).success(
        bot, [gpt_msg, gemini_msg, chat_1, chat_2], [chat_1, chat_2])

    btn.click(init, [state, text_msg, image], [state, sys_msg], queue=False).then(
        raise_exception, sys_msg, None).success(
        user, [text_msg, chat_1, chat_2], [text_msg, chat_1, chat_2], queue=False).success(
        get_response, [state, image, chat_1, chat_2],  [gpt_msg, gemini_msg, image, sys_msg, chat_1, chat_2]).success(
        bot, [gpt_msg, gemini_msg, chat_1, chat_2], [chat_1, chat_2])

    # クリア時でもセッションの一部は残す
    btn_clear.click(clear_click, state, state)

    with gr.Tab("注意事項") as notes:
            caution = '・GPTのモデルは"gpt-4-1106-preview"です。<br>'
            caution += "・設定タブからモデルをGPT-3.5に変更できます。<br>"
            caution += "・こういうときにエラーになるなどフィードバックあればお待ちしています。"
            gr.Markdown("<h3>" + caution + "</h3>")


if __name__ == '__main__':

    demo.queue()
    demo.launch(debug=True)