import { useEffect, useRef, useState } from "react";
import "./App.css";
import { clef, musicNote, pen } from "./assets/svg";
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
    if (forwardMove.current || backwardMove.current) return;

    console.log("enter");

    forwardMove.current = true;

    let originalText = (translationText + "    ").split("");
    let newTranslationText = "Translate".split("");

    const penSVG = translationButtonRef.current.children[0];

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
    if (forwardMove.current || backwardMove.current) return;

    console.log("leave");

    backwardMove.current = true;

    const penSVG = translationButtonRef.current.children[0];

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

          <button
            onMouseEnter={enterAnimation}
            onMouseLeave={leaveAnimation}
            ref={translationButtonRef}
            className="button translation"
          >
            {pen}
            <span ref={translationButtonTextRef}>{translationText}</span>
          </button>
          {/* <button className="button convert">تحويل</button> */}
        </div>
      </div>
    </>
  );
}

export default App;
