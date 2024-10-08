import { useEffect, useRef, useState } from "react";
import * as Tone from "tone";
import Typewriter from "typewriter-effect";
import "./App.css";
import Stick from "./assets/Group.svg";
import Phonograph from "./assets/Group2.svg";
import { clef, maracas, musicNote, pen } from "./assets/svg";
import "./Buttons.css";
import CursorAnimation from "./CursorAnimation";
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
  const [isRecording, setIsRecording] = useState(false); //? Check if the recording is active
  const [recorder, setRecorder] = useState(null); //? The MediaRecorder object
  const [audioBlob, setAudioBlob] = useState(null); //? Store the audio as Blob

  const [arabicText, setArabicText] = useState(null); //? The Arabic text
  const [englishText, setEnglishText] = useState(null); //? The English text

  const startRecording = async () => {
    setAsrResult(null);
    setAudioURL(null);
    setAudioBlob(null);
    setSpeech2Text(false);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      const audioBlob = new Blob([event.data], {
        type: "audio/wav; codecs=MS_PCM",
      });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioURL(audioUrl);
      setAudioBlob(audioBlob);
    };
    mediaRecorder.start();
    setRecorder(mediaRecorder);
    setIsRecording(true);
  };

  const stopRecording = () => {
    recorder.stop();
    setIsRecording(false);
  };

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

  useEffect(() => {
    // Clean up: stop the recorder if the component is unmounted
    return () => {
      if (recorder) {
        recorder.stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [recorder]);

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
    // sampler.triggerAttackRelease(note, "8n");
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
        <div className="content">
          <div className="header">
            <img
              className="small-image hafz"
              src="./src/assets/abd elhaleem hafz.jpg"
              alt=""
            />
            <img
              className="small-image faried"
              src="./src/assets/faried.jpg"
              alt=""
            />
            <img
              className="small-image big-boss"
              src="./src/assets/big boss.jpg"
              alt=""
            />
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
              {!isRecording && (
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
              )}

              {/* <div className="recording">
                <div className={`circle1 ${isRecording ? "pulse1" : ""}`}></div>
                <div className={`circle2 ${isRecording ? "pulse2" : ""}`}></div>
                <div
                  className={`mic ${isRecording ? "pulse3" : ""}`}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="-0.5 -0.5 16 16"
                    id="Stop-Fill--Streamline-Mingcute-Fill"
                    height={16}
                    width={16}
                    className={isRecording ? "" : "hide"}
                  >
                    <desc>
                      {"Stop Fill Streamline Icon: https://streamlinehq.com"}
                    </desc>
                    <g fill="none" fillRule="evenodd">
                      <path
                        fill="#ffffff"
                        d="M2.5 3.75a1.25 1.25 0 0 1 1.25 -1.25h7.5a1.25 1.25 0 0 1 1.25 1.25v7.5a1.25 1.25 0 0 1 -1.25 1.25H3.75a1.25 1.25 0 0 1 -1.25 -1.25V3.75Z"
                        strokeWidth={1}
                      />
                    </g>
                  </svg>

                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="-0.5 -0.5 16 16"
                    id="Mic-Fill--Streamline-Mingcute-Fill"
                    height={16}
                    width={16}
                    className={!isRecording ? "" : "hide"}
                  >
                    <desc>
                      {"Mic Fill Streamline Icon: https://streamlinehq.com"}
                    </desc>
                    <g fill="none" fillRule="nonzero">
                      <path
                        fill="#ffffff"
                        d="M11.91875 7.50625a0.625 0.625 0 0 1 0.53125 0.7074999999999999A5.0024999999999995 5.0024999999999995 0 0 1 8.125 12.46125V13.125a0.625 0.625 0 1 1 -1.25 0v-0.6637500000000001a5.003125000000001 5.003125000000001 0 0 1 -4.324375 -4.2475000000000005 0.625 0.625 0 0 1 1.2375 -0.1775 3.7506250000000003 3.7506250000000003 0 0 0 7.42375 0 0.625 0.625 0 0 1 0.7074999999999999 -0.53ZM7.5 1.25a3.125 3.125 0 0 1 3.125 3.125v3.125a3.125 3.125 0 0 1 -6.25 0V4.375a3.125 3.125 0 0 1 3.125 -3.125Z"
                        strokeWidth={1}
                      />
                    </g>
                  </svg>
                </div>
              </div> */}
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
                  <img
                    className="drum-png"
                    src="./src/assets/drum.png"
                    alt=""
                  />
                  <img
                    className="trumpet-png"
                    src="./src/assets/trumpet.png"
                    alt=""
                  />
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
