import { useEffect, useRef, useState } from "react";
import Typewriter from "typewriter-effect";
import { sendAudio, sendText } from "./API";
import "./App.css";
import Stick from "./assets/Group.svg";
import Phonograph from "./assets/Group2.svg";
import {
  backArrow,
  clef,
  github,
  maracas,
  musicNote,
  pen,
  upArrow,
} from "./assets/svg";
import "./Buttons.css";
import CopyButton from "./CopyButton";
import CursorAnimation from "./CursorAnimation";
import "./Header.css";
import "./loader.css";
import "./MagicButton.css";
import "./Translation.css";
import { enterAnimation, leaveAnimation } from "./translationAnimation";
import { playNote, startAudioContext } from "./utils";

function App() {
  CursorAnimation();

  const [translationActive, setTranslationActive] = useState(false); //? check if we are in the translation section
  const [asrActive, setAsrActive] = useState(false); //? check if we are in the ASR section
  const [speech2text, setSpeech2Text] = useState(false); //?check if the button is clicked to convert speech to text
  const [asrResult, setAsrResult] = useState(null); //? The result of the speech to text

  const translationButtonTextRef = useRef(null);
  const translationButtonRef = useRef(null);
  const [translationText, setTranslationText] = useState("ترجمة"); //? The text of the translation button
  const forwardMove = useRef(false); //? boolean to check if the forward animation is running
  const backwardMove = useRef(false); //? boolean to check if the backward animation is running

  const [audioURL, setAudioURL] = useState(""); //? The URL of the recorded audio
  const [audioBlob, setAudioBlob] = useState(null); //? Store the audio as Blob

  const [arabicText, setArabicText] = useState(null); //? The Arabic text
  const [englishText, setEnglishText] = useState(null); //? The English text
  const [translationTitle, setTranslationTitle] = useState("English"); //? The title of the translation section
  const [isChecked, setIsChecked] = useState(false);

  useEffect(() => {
    startAudioContext();
  }, []);

  const handelBackButton = () => {
    setAsrActive(false);
    setTranslationActive(false);
    setEnglishText("");
  };

  const backButton = (colorClassName) => {
    return (
      <div
        className={`back-button ${colorClassName}`}
        onClick={handelBackButton}
      >
        {backArrow}
      </div>
    );
  };

  const handleCheckboxChange = (event) => {
    const checked = event.target.checked;
    setIsChecked(checked);
    setTranslationTitle(checked ? "Auto Detection" : "English");
  };

  return (
    <>
      <div className="main">
        {clef(playNote)} {musicNote(playNote)}
        <img src={Phonograph} className="phonograph" alt="Group Icon" />
        <img src={Stick} className="stick" alt="Group Icon" />
        <div className="light-2"></div>
        <div className={`navbar`}>
          <a
            className="org"
            href="https://github.com/marwan2232004/Esma3ny"
            target="_blank noreferrer"
          >
            <div className="icon">
              <button className="Btn">
                <span className="svgContainer">{github}</span>
                <span className="BG"></span>
              </button>
            </div>
          </a>
        </div>
        <div className="content">
          <div className="header">
            <img className="small-image hafz" src="/hafz.jpg" alt="" />
            <img className="small-image faried" src="/faried.jpg" alt="" />
            <img className="small-image big-boss" src="/big boss.jpg" alt="" />
            <div className="name">اسمعني</div>
          </div>

          {/* //? Description section */}

          {!asrActive && !translationActive && (
            <div className="description ">
              {` موقع "اسمعني" هو أداة مبتكرة تقوم بتحويل الصوت إلى نص بسهولة وسرعة.
            يتيح لك الموقع الحصول على النص إما بنفس اللغة التي تم التحدث بها أو
            ترجمته إلى لغة أخرى، مما يسهل التواصل وتوفير الوقت في الكتابة أو
            الترجمة.`}
            </div>
          )}

          {/* //? Upload and Record Audio Buttons section */}

          {asrActive && (
            <div className="asr">
              {
                <>
                  <button className={` button `}>
                    <input
                      className="audio-input"
                      type="file"
                      accept="audio/*"
                      onChange={(e) => {
                        const file = e.target.files[0];
                        if (file) {
                          // Create a Blob from the selected file
                          const blob = new Blob([file], { type: "audio/wav" });
                          console.log(blob);
                          setAudioBlob(blob);

                          // Create a Blob URL for the audio file
                          const fileURL = URL.createObjectURL(blob);
                          setAudioURL(fileURL);
                          setAsrResult(null);
                          setSpeech2Text(false);
                        }
                        e.target.value = null; // Reset the input value
                      }}
                    />
                    {upArrow}
                  </button>
                </>
              }
            </div>
          )}

          {asrActive && !audioURL && <div style={{ height: "3rem" }}></div>}

          {/* //? Display the audio player if the audio is recorded and the ASR is active */}

          {asrActive && audioURL && (
            <div
              style={{
                marginBottom: "2rem",
              }}
            >
              <audio src={audioURL} controls />
            </div>
          )}

          {/* //? Loading until the result of the asr is fetched */}

          {speech2text && asrActive && !asrResult && (
            <div className="loader"></div>
          )}

          {/* //? Typewriter effect to type the result of the speech to text */}
          {asrActive && asrResult && (
            <div className="asr-result-showcase">
              <div
                style={{ position: "absolute", top: "-5rem", right: "5rem" }}
              >
                <CopyButton text={asrResult} />
              </div>

              <div
                style={{
                  fontSize: "26px",
                  fontWeight: "bold",
                  color: "white",
                  width: "90rem",
                  wordWrap: "break-word",
                  marginBottom: "2rem",
                  userSelect: "text",
                }}
              >
                <Typewriter
                  options={{
                    delay: 90, // Lower delay for faster typing
                  }}
                  onInit={(typewriter) => {
                    typewriter.typeString(asrResult).pauseFor(2500).start();
                  }}
                />
              </div>
            </div>
          )}

          {translationActive && (
            <div className="translation-section">
              <div className="english-section">
                <textarea
                  value={englishText}
                  dir="ltr"
                  onInput={(e) => {
                    let value = e.target.value;
                    if (!isChecked) {
                      value = value.replace(/[^a-zA-Z0-9\s.,;:?!]/g, "");
                    }
                    setEnglishText(value);
                  }}
                ></textarea>
                <div className="english-header">
                  <div className="english-title">{translationTitle}</div>
                  <div className="checkbox-wrapper-5">
                    <div className="check">
                      <input
                        checked={isChecked}
                        id="check-5"
                        type="checkbox"
                        onChange={handleCheckboxChange}
                      />
                      <label htmlFor="check-5"></label>
                    </div>
                  </div>
                </div>
              </div>

              <div className="arabic-section">
                <div className="arabic-header">
                  <div className="arabic-title">{`عربى`}</div>
                  <CopyButton text={arabicText} />
                </div>
                <textarea
                  readOnly
                  name="arabic"
                  id="arabic"
                  dir="rtl"
                  value={arabicText}
                ></textarea>
              </div>
            </div>
          )}

          {/* //? Buttons section */}

          <div className="buttons">
            {!translationActive && (
              <div className="group">
                <div
                  className="button-container"
                  onClick={async () => {
                    if (audioBlob && asrActive) {
                      setSpeech2Text(true);
                      const res = await sendAudio({
                        audioBlob,
                        setResult: setAsrResult,
                      });
                      if (!res) {
                        setSpeech2Text(false);
                      }
                    } else setSpeech2Text(false);
                    setAsrActive(true);
                  }}
                >
                  <img className="drum-png" src="/drum.png" alt="" />
                  <img className="trumpet-png" src="/trumpet.png" alt="" />
                  {maracas}
                  <button className="button speak">
                    {!asrActive ? "تحدث" : "تحويل"}
                  </button>
                </div>
                {asrActive && backButton("asr-back-color")}
              </div>
            )}

            {!asrActive && (
              <div className="group">
                <button
                  onMouseEnter={() =>
                    enterAnimation({
                      forwardMove,
                      backwardMove,
                      translationButtonRef,
                      translationText,
                      setTranslationText,
                    })
                  }
                  onMouseLeave={() =>
                    leaveAnimation({
                      forwardMove,
                      backwardMove,
                      translationButtonRef,
                      translationText,
                      setTranslationText,
                    })
                  }
                  ref={translationButtonRef}
                  className="button translation"
                  onClick={() => {
                    if (translationActive && englishText) {
                      sendText({
                        englishText,
                        isChecked,
                        setResult: setArabicText,
                      });
                    }
                    setTranslationActive(true);
                  }}
                >
                  {pen}
                  <span ref={translationButtonTextRef}>{translationText}</span>
                </button>
                {translationActive && backButton("translation-back-color")}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
