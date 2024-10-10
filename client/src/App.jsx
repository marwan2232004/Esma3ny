import { useRef, useState } from "react";
import * as Tone from "tone";
import Typewriter from "typewriter-effect";
import "./App.css";
import Stick from "./assets/Group.svg";
import Phonograph from "./assets/Group2.svg";
import { clef, maracas, musicNote, pen } from "./assets/svg";
import "./Buttons.css";
import CursorAnimation from "./CursorAnimation";
import "./Header.css";
import "./loader.css";
import "./Translation.css";

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

  const sendAudioToAPI = async () => {
    if (!audioBlob) return;

    console.log(audioBlob);
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.wav"); // Append the Blob as a file
    try {
      const response = await fetch("http://127.0.0.1:8000/audio2text", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log(result);
        setAsrResult(result.text);
        setAudioURL(null);
        setAudioBlob(null);
      }
      setSpeech2Text(false);
    } catch (error) {
      console.error("Error sending audio:", error);
    }
  };

  const sendTextToAPI = async () => {
    if (!arabicText) return;
    console.log(JSON.stringify({ text: arabicText }));
    try {
      const response = await fetch("http://127.0.0.1:8000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: arabicText }),
      });

      if (response.ok) {
        console.log("Text translated successfully!");
        const result = await response.json();
        setEnglishText(result.translation);
      } else {
        console.error("Failed to upload audio");
      }
    } catch (error) {
      console.error("Error sending audio:", error);
    }
  };

  const enterAnimation = () => {
    //? To prevent the animation from running multiple times
    if (
      forwardMove.current ||
      backwardMove.current ||
      translationText === "Translate"
    )
      return;

    forwardMove.current = true;

    let originalText = (translationText + "    ").split("");
    let newTranslationText = "Translate".split("");

    const penSVG = translationButtonRef.current.children[0];

    //?modify the pen position so that when removing the animation it do not return to original position
    //?then adding the backward animation
    penSVG.style.left = "-0.5rem";
    penSVG.style.top = "-0.5rem";
    penSVG.style.transform = "scale(-1, 1)";
    penSVG.classList.add("forward-move");
    penSVG.classList.remove("backward-move");

    setTimeout(() => {
      for (let i = 0; i < 9; i++) {
        setTimeout(() => {
          originalText[i] = newTranslationText[i];
          setTranslationText(originalText.join(""));
          if (i == 8) forwardMove.current = false;
        }, 70 * i);
      }
    }, 800);
  };

  const leaveAnimation = () => {
    //? To prevent the animation from running multiple times
    if (
      forwardMove.current ||
      backwardMove.current ||
      translationText === "ترجمه"
    )
      return;

    backwardMove.current = true;

    const penSVG = translationButtonRef.current.children[0];

    //?modify the pen position so that when removing the animation it do not return to original position
    //?then adding the backward animation
    penSVG.style.left = "7rem";
    penSVG.style.top = "-0.5rem";
    penSVG.style.transform = "scale(1, 1)";
    penSVG.classList.remove("forward-move");
    penSVG.classList.add("backward-move");

    let originalText = translationText.split("");
    let newTranslationText = "ترجمة".split("");

    setTimeout(() => {
      for (let i = 0; i < 9; i++) {
        setTimeout(() => {
          if (i < newTranslationText.length)
            originalText[i] = newTranslationText[i];
          else originalText[i] = "";
          setTranslationText(originalText.join(""));
          if (i == 8) backwardMove.current = false;
        }, 70 * i);
      }
    }, 800);
  };
  const [audioContextStarted, setAudioContextStarted] = useState(false);

  const playNote = (note) => {
    startAudioContext();
    const sampler = new Tone.Sampler({
      urls: {
        A1: "A1.mp3",
        A2: "A2.mp3",
      },
      baseUrl: "https://tonejs.github.io/audio/casio/",
      onload: () => {
        sampler.triggerAttackRelease([note], 0.5);
      },
    }).toDestination();
  };

  // Function to start the Tone.js audio context
  const startAudioContext = async () => {
    if (!audioContextStarted) {
      await Tone.start();
      setAudioContextStarted(true);
      console.log("Audio Context Started");
    }
  };

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
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512">
          <path
            fill="#ffffff"
            d="M9.4 233.4c-12.5 12.5-12.5 32.8 0 45.3l160 160c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3L109.2 288 416 288c17.7 0 32-14.3 32-32s-14.3-32-32-32l-306.7 0L214.6 118.6c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0l-160 160z"
          />
        </svg>
      </div>
    );
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
            href="https://github.com/marwan2232004"
            target="_blank noreferrer"
          >
            <div className="icon">
              <button className="Btn">
                <span className="svgContainer">
                  <svg fill="white" viewBox="0 0 496 512">
                    <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"></path>
                  </svg>
                </span>
                <span className="BG"></span>
              </button>
            </div>
          </a>
        </div>
        <div className="content">
          <div className="header">
            <img
              className="small-image hafz"
              src="/abd elhaleem hafz.jpg"
              alt=""
            />
            <img className="small-image faried" src="/faried.jpg" alt="" />
            <img className="small-image big-boss" src="/big boss.jpg" alt="" />
            <div className="name">أسمعنى</div>
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
                      accept="audio/*" // Only allow audio file selection
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
                        }
                        e.target.value = null; // Reset the input value
                      }}
                    />
                    <svg className="svgIcon" viewBox="0 0 384 512">
                      <path d="M214.6 41.4c-12.5-12.5-32.8-12.5-45.3 0l-160 160c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L160 141.2V448c0 17.7 14.3 32 32 32s32-14.3 32-32V141.2L329.4 246.6c12.5 12.5 32.8 12.5 45.3 0s12.5-32.8 0-45.3l-160-160z"></path>
                    </svg>
                  </button>
                </>
              }
            </div>
          )}

          {/* //? Loading until the result of the asr is fetched */}

          {speech2text && asrActive && !asrResult && (
            <div className="loader"></div>
          )}

          {/* //? Typewriter effect to type the result of the speech to text */}
          {asrActive && asrResult && (
            <div
              style={{
                fontSize: "26px",
                fontWeight: "bold",
                color: "white",
                width: "90rem", // Set width to 90rem
                wordWrap: "break-word", // Enable text wrapping
                marginBottom: "2rem",
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
          )}

          {/* //? Display the audio player if the audio is recorded and the ASR is active */}

          {asrActive && audioURL && !speech2text && (
            <div>
              <audio src={audioURL} controls />
            </div>
          )}

          {translationActive && (
            <div className="translation-section">
              <div className="arabic-section">
                <div className="arabic-title">{`عربى`}</div>
                <textarea
                  name="arabic"
                  id="arabic"
                  dir="rtl"
                  //? Allow Arabic characters, numbers, and punctuation
                  onInput={(e) => {
                    const value = e.target.value;
                    const arabicRegex = /^[\u0600-\u06FF0-9\s.,;:?!،؛؟]*$/;
                    if (!arabicRegex.test(value)) {
                      e.target.value = value.replace(
                        /[^\u0600-\u06FF0-9\s.,;:?!،؛؟]/g,
                        ""
                      );
                    }
                    setArabicText(e.target.value);
                  }}
                ></textarea>
              </div>

              <div className="english-section">
                <div className="english-title">{`English`}</div>
                <textarea
                  readOnly
                  name="english"
                  id="english"
                  value={englishText}
                ></textarea>
              </div>
            </div>
          )}

          <div className="buttons">
            {!translationActive && (
              <div className="group">
                <div
                  className="button-container"
                  onClick={() => {
                    if (audioBlob && asrActive) {
                      setSpeech2Text(true);
                      sendAudioToAPI();
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
                  onMouseEnter={enterAnimation}
                  onMouseLeave={leaveAnimation}
                  ref={translationButtonRef}
                  className="button translation"
                  onClick={() => {
                    if (translationActive && arabicText) {
                      sendTextToAPI();
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
