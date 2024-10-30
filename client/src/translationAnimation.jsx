const enterAnimation = ({
  forwardMove,
  backwardMove,
  translationButtonRef,
  translationText,
  setTranslationText,
}) => {
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

const leaveAnimation = ({
  forwardMove,
  backwardMove,
  translationButtonRef,
  translationText,
  setTranslationText,
}) => {
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

export { enterAnimation, leaveAnimation };
