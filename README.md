# 上海闲话Anki制作工具 Shanghainese Anki Deck Builder (ZHAB)

This is a Python script for converting a list of Shanghainese sentences into Wugniu romanisations and high-quality TTS (Text-to-Speech) audio for spaced repetition.

The program produced mainly uses Simplified Chinese and was tested with it, as this is the script mainly used in Shanghai. However, Unihan is fully supported and Traditional Chinese texts can be used with no problem. 

## Pipeline
`sentences.csv -> romanisation -> TTS audio (WAV/MP3) -> Anki-ready media`

## Structure
```
ZHAB/
├─ zhab.py
├─ README.md
├─ requirements.txt
├─ sentences.csv
├─ extract_900ju_pdf
|	├─900-sentences.pdf # Test dataset
|	├─extract_pdf.py
|	├─sentences.csv
└─ out/
	├─audio
	├─gradio-tmp
	├─failed.csv
	├─Shanghainese TTS (HF).apkg
	├─state.json
	└─yahwe_zaonhe.dict.yaml
```

## Usage
`bash python zhab.py`

There are several internal config options the user ought to be aware of;
```
MAX_ATTEMPTS_PER_ROW = 4
BASE_BACKOFF_SECONDS = 2.0
JITTER_SECONDS = 0.6
FAILED_QUEUE_PASSES = 6
FAILED_PASS_PAUSE_SECONDS = 20.0
RETRY_FAILED_EVERY_N_SUCCESSES = 50  # 0 disables mid-run retry

OUT_DIR = Path("out")
AUDIO_DIR = OUT_DIR / "audio"
STATE_PATH = OUT_DIR / "state.json"
FAILED_PATH = OUT_DIR / "failed.csv"
RIME_CACHE_PATH = OUT_DIR / "yahwe_zaonhe.dict.yaml"

MODEL_ID = 190101001
DECK_ID = 190101002

BRACKET_UNKNOWN = True
KEEP_WAV = True
PRINT_ROMANISATION = True
```

You should see something like this:
```
[OK  775] 如果我辰光有钞票够，我要到欧洲去旅游。
          → zyu ku ngu/ngu zen kuaon yeu [钞] phiau [够]， ngu/ngu iau tau [欧] tseu chi/chiu li/liu yeu。
          → zh: 如果我有时间有钱，我要到欧洲去旅游。
          → audio: shh_aad3278cf4fd88af.mp3
[DEBUG] predict() return type=<class 'dict'> value={'name': '/tmp/tmpmbv64vry/tmpr66hmmol.wav', 'data': None, 'is_file': True}
[DEBUG] saved audio to: out\audio\shh_e2311b3be02221b4.wav exists=True
[mp3] ffmpeg input: out\audio\shh_e2311b3be02221b4.wav
[OK  776] 亨八冷打我搿趟旅程需要十二天。
          → han paeh lan tan ngu/ngu geh thaon li/liu zen siu iau zeh nyi thie。
          → zh: 我这趟旅程一共在内需要十二天。
          → audio: shh_e2311b3be02221b4.mp3
[DEBUG] predict() return type=<class 'dict'> value={'name': '/tmp/tmpmbv64vry/tmp_glpa39v.wav', 'data': None, 'is_file': True}
[DEBUG] saved audio to: out\audio\shh_e4fa740013e991b0.wav exists=True
Generating:  93%|█████████████████████████████████████████████████████████████▉     | 803/868 [37:55<03:27,  3.19s/row]
```

## Dependencies 
```
pandas>=1.5
requests>=2.28
gradio-client>=0.10
genanki>=0.13
tqdm>=4.65
```

Use:
`pip install -r requirements.txt`

You also need to install ffmpeg (e.g. `winget install Gyan.FFmpeg`) and add it to PATH. 

## Notes
It assumes you have a `sentences.csv`. Edit the provided one for best results; it uses a set of 900 sample sentences with Mandarin meanings. 

This does not output English translations, but if you have OmniPrompt on Anki with a DeepSeek API key you should be able to get something half-decent out of it. If you're crazy enough to learn Shanghainese, I would not be shocked if you have that.

TTS is derived from this Hugging Face space:
https://huggingface.co/spaces/CjangCjengh/Shanghainese-TTS

The TTS is good but the tone sandhi can be wonky sometimes (it changes on syntax, grammar, and context, so who can blame it?). It generally takes ~3sec per row for speech to generate. 

Romanisation is derived from the following Rime dictionary and is one of the best publicly available resources for Shanghainese:
https://raw.githubusercontent.com/wugniu/rime-yahwe_zaonhe/master/yahwe_zaonhe.dict.yaml

It is also a good rime input system that I encourage you to use while learning Shanghainese. Any unknown characters in this script's romanisation are marked with square brackets: `[字]`.

## Limitations
The decks this program produces could be improved through adding English translations; DeepSeek API with OmniPrompt could do this with great success and speed. As I have access to this feature myself, I chose not to add it to the program (I made this for me, after all!). Additionally, it does not robustly use the functions Anki has (e.g. Cloze detection, etc). 

The TTS program does use a speed function; a savvy user could change the spreadsheet to use random speeds for different levels of comprehension. I kept it to 1.0, as I did not extensively use the feature. 

This only supports Shanghainese, but could be tweaked to work with other Chinese topolects through minor tweaks here and there (particularly, having a good rime dict and TTS space). However, accessing the Gradio client was a small nightmare, so some work would be needed to get rid of all the hacky code I used. 

## Acknowledgements
ChatGPT was used to assist the development and debugging of this program. 

## References
- Chen, Y. (2022). wugniu/rime-yahwe_zaonhe: 吳語協會式上海話輸入法/吴语协会式上海话输入法：以吳語協會式拼音爲基礎的 Rime 上海話輸入方案。 (Version 1.0) [Windows, Mac, Linux]. Wugniu Yaqdaon. https://github.com/wugniu/rime-yahwe_zaonhe
- CjangCjengh. (2022). Shanghainese TTS—A Hugging Face Space by CjangCjengh (Version 3.4.1) [Python]. https://huggingface.co/spaces/CjangCjengh/Shanghainese-TTS

## Licence
As a supporter of the [Free Software Movement](https://www.fsf.org/about/) and its values, this is published under a [GNU General Public Licence v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). Contributions are encouraged. 
