import { useEffect, useRef, useState } from "react";
import "./App.css";
import { clef, maracas, musicNote, pen } from "./assets/svg";
import "./Buttons.css";
import CursorAnimation from "./CursorAnimation";

function App() {
  CursorAnimation();

  const translationButtonTextRef = useRef(null);
  const translationButtonRef = useRef(null);
  const [translationText, setTranslationText] = useState("ترجمة");
  const forwardMove = useRef(false);
  const backwardMove = useRef(false);

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

  return (
    <>
      <div className="main">
        {clef} {musicNote} <div className="light-1"></div>{" "}
        <div className="light-2"></div>{" "}
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

          <div className="description">
            موقع "اسمعني" هو أداة مبتكرة تقوم بتحويل الصوت إلى نص بسهولة وسرعة.
            يتيح لك الموقع الحصول على النص إما بنفس اللغة التي تم التحدث بها أو
            ترجمته إلى لغة أخرى، مما يسهل التواصل وتوفير الوقت في الكتابة أو
            الترجمة.
          </div>

          <div className="buttons">
            <div className="button-container">
              <img className="drum-png" src="./src/assets/drum.png" alt="" />
              <img
                className="trumpet-png"
                src="./src/assets/trumpet.png"
                alt=""
              />
              {maracas}
              <button className="button speak">تحدث</button>
            </div>
            <button
              onMouseEnter={enterAnimation}
              onMouseLeave={leaveAnimation}
              ref={translationButtonRef}
              className="button translation"
            >
              {pen}
              <span ref={translationButtonTextRef}>{translationText}</span>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
