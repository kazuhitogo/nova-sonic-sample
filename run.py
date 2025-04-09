import boto3
import os
import asyncio
import base64
import json
import uuid
import pyaudio

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import (
    Config,
    HTTPAuthSchemeResolver,
    SigV4AuthScheme,
)
from smithy_aws_core.credentials_resolvers.environment import (
    EnvironmentCredentialsResolver,
)

# 音声設定
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024


class NovaVoiceChat:
    def __init__(
        self, model_id='amazon.nova-sonic-v1:0', region='us-east-1', voice_id='matthew'
    ):
        self.model_id = model_id
        self.region = region
        self.voice_id = voice_id
        self.client = None
        self.stream = None
        self.response = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue = asyncio.Queue()
        self.display_assistant_text = False
        self.role = None

    def _initialize_client(self):
        """Bedrockクライアントを初期化"""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.client = BedrockRuntimeClient(config=config)

    async def send_event(self, event_json):
        """イベントをストリームに送信"""
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        """Nova Sonicとのセッションを開始"""
        if not self.client:
            self._initialize_client()

        # ストリームを初期化
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        # セッション開始イベントを送信
        session_start = '''
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        '''
        await self.send_event(session_start)

        # プロンプト開始イベントを送信
        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self.prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "{self.voice_id}",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        '''
        await self.send_event(prompt_start)

        # システムプロンプトを送信
        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "type": "TEXT",
                    "interactive": true,
                    "role": "SYSTEM",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(text_content_start)

        system_prompt = (
            "You are a friendly assistant. The user and you will engage in a spoken dialog "
            "exchanging the transcripts of a natural real-time conversation. Keep your responses short, "
            "generally two or three sentences for chatty scenarios."
        )

        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "content": "{system_prompt}"
                }}
            }}
        }}
        '''
        await self.send_event(text_input)

        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(text_content_end)

        # レスポンス処理を開始
        self.response = asyncio.create_task(self._process_responses())

    async def start_audio_input(self):
        """音声入力ストリームを開始"""
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_start)

    async def send_audio_chunk(self, audio_bytes):
        """音声チャンクをストリームに送信"""
        if not self.is_active:
            return

        blob = base64.b64encode(audio_bytes)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "content": "{blob.decode('utf-8')}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_event)

    async def end_audio_input(self):
        """音声入力ストリームを終了"""
        audio_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_end)

    async def end_session(self):
        """セッションを終了"""
        if not self.is_active:
            return

        prompt_end = f'''
        {{
            "event": {{
                "promptEnd": {{
                    "promptName": "{self.prompt_name}"
                }}
            }}
        }}
        '''
        await self.send_event(prompt_end)

        session_end = '''
        {
            "event": {
                "sessionEnd": {}
            }
        }
        '''
        await self.send_event(session_end)
        # ストリームを閉じる
        await self.stream.input_stream.close()
        self.is_active = False

    async def _process_responses(self):
        """ストリームからのレスポンスを処理"""
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)

                    if 'event' in json_data:
                        # コンテンツ開始イベントを処理
                        if 'contentStart' in json_data['event']:
                            content_start = json_data['event']['contentStart']
                            # ロールを設定
                            self.role = content_start['role']
                            # 推測内容を確認
                            if 'additionalModelFields' in content_start:
                                additional_fields = json.loads(
                                    content_start['additionalModelFields']
                                )
                                if (
                                    additional_fields.get('generationStage')
                                    == 'SPECULATIVE'
                                ):
                                    self.display_assistant_text = True
                                else:
                                    self.display_assistant_text = False

                        # テキスト出力イベントを処理
                        elif 'textOutput' in json_data['event']:
                            text = json_data['event']['textOutput']['content']

                            if self.role == "ASSISTANT" and self.display_assistant_text:
                                print(f"アシスタント: {text}")
                            elif self.role == "USER":
                                print(f"ユーザー: {text}")

                        # 音声出力を処理
                        elif 'audioOutput' in json_data['event']:
                            audio_content = json_data['event']['audioOutput']['content']
                            audio_bytes = base64.b64decode(audio_content)
                            await self.audio_queue.put(audio_bytes)
        except asyncio.CancelledError:
            print("レスポンス処理がキャンセルされました")
        except Exception as e:
            print(f"レスポンス処理エラー: {e}")

    async def play_audio(self):
        """音声レスポンスを再生"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT, channels=CHANNELS, rate=OUTPUT_SAMPLE_RATE, output=True
        )

        try:
            while self.is_active:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=0.5
                    )
                    stream.write(audio_data)
                except asyncio.TimeoutError:
                    # 音声データがない場合は続行
                    continue
        except asyncio.CancelledError:
            print("音声再生がキャンセルされました")
        except Exception as e:
            print(f"音声再生エラー: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def capture_audio(self):
        """マイクから音声をキャプチャしてNova Sonicに送信"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        print("マイクに向かって話してください...")
        print("終了するにはEnterキーを押してください...")

        await self.start_audio_input()

        try:
            while self.is_active:
                audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("音声キャプチャがキャンセルされました")
        except Exception as e:
            print(f"音声キャプチャエラー: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("音声キャプチャを停止しました")
            await self.end_audio_input()


async def main():
    # 利用可能な音声を表示
    print("利用可能な音声:")
    print("1. tiffany (女性)")
    print("2. matthew (男性)")
    print("3. amy (女性)")
    # ユーザーに音声を選択させる
    voice_choice = input("音声を選択してください (1-3) [デフォルト: tiffany]: ")

    # 選択に基づいて音声IDを設定
    voices = {
        "1": "tiffany",
        "2": "matthew",
        "3": "amy",
    }

    voice_id = voices.get(voice_choice, "tiffany")

    print(f"選択された音声: {voice_id}")

    # Nova Sonicクライアントを作成
    nova_client = NovaVoiceChat(voice_id=voice_id)

    # セッションを開始
    print("セッションを開始しています...")
    await nova_client.start_session()

    # 音声再生タスクを開始
    playback_task = asyncio.create_task(nova_client.play_audio())

    # 音声キャプチャタスクを開始
    capture_task = asyncio.create_task(nova_client.capture_audio())

    try:
        # ユーザーがEnterを押すのを待つ
        await asyncio.get_event_loop().run_in_executor(None, input)
    finally:
        # セッションを終了
        print("セッションを終了しています...")
        nova_client.is_active = False

        # タスクをキャンセル
        tasks = []
        if not playback_task.done():
            tasks.append(playback_task)
        if not capture_task.done():
            tasks.append(capture_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # レスポンスタスクをキャンセル
        if nova_client.response and not nova_client.response.done():
            nova_client.response.cancel()

        await nova_client.end_session()
        print("セッションが正常に終了しました")


if __name__ == "__main__":
    session = boto3.Session()
    credentials = session.get_credentials()
    os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key
    os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("プログラムが中断されました")
