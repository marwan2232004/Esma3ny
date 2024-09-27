import { useEffect, useRef } from "react";
import "./cursorAnimation.css";

function CursorAnimation() {
  const animationCount = useRef(0); // Use useRef to keep track of animation count
  const distanceThreshold = 80;
  const iconIntervalTime = 2000;
  const glowIntervalTime = 50;

  const CalcDistance = (x1, y1, x2, y2) => {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
  };
  const colors = ["#ff0478", "#afa7e6", "#E6B6A7"];
  const icons = [
    ` <desc>{"Music Note 2 Streamline Icon: https://streamlinehq.com"}</desc>
        <path d="M8.18375 7.5V0.6628125" stroke=${colors[0]} strokeLinecap="round" strokeWidth="1"/>
        <path stroke=${colors[0]} d="M2.7139375 11.6023125C2.7139375 13.707625 4.993 15.0234375 6.8163125 13.970812500000001C7.6624375 13.48225 8.18375 12.579374999999999 8.18375 11.6023125C8.18375 9.497 5.904625 8.1811875 4.0813749999999995 9.233875000000001C3.23525 9.722375 2.7139375 10.62525 2.7139375 11.6023125" strokeWidth="1"/>
        <path d="M12.2860625 4.765125C10.020375 4.765125 8.18375 2.9284375000000002 8.18375 0.6628125" stroke=${colors[0]} strokeLinecap="round" strokeWidth="1"/>`,
    `<desc>{"Music Notes Simple Bold Streamline Icon: https://streamlinehq.com"}</desc><path stroke=${colors[1]} strokeLinecap="round" strokeWidth="1"  d="M14.021472656250001 0.31655859374999995c-0.191080078125 -0.14904492187499999 -0.44014453125 -0.20180859375 -0.67524609375 -0.143056640625L4.946671875 2.2733906249999998C4.59615234375 2.361005859375 4.350228515625 2.67591796875 4.35017578125 3.037224609375v6.7819804687500005c-1.8519316406250002 -0.8082421875 -3.884337890625 0.69137109375 -3.6583359375 2.6993085937499997 0.226001953125 2.007943359375 2.540912109375 3.018251953125 4.1668359375 1.818556640625 0.670822265625 -0.49497070312499997 1.0666347656250001 -1.2791718749999998 1.0664121093749999 -2.1128378906250003V3.6521015625l6.82463671875 -1.7061621093750001v5.774689453125c-1.85192578125 -0.8082421875 -3.8843320312499996 0.69137109375 -3.6583300781249997 2.699314453125 0.226001953125 2.0079374999999997 2.540912109375 3.01824609375 4.1668359375 1.81855078125 0.671197265625 -0.495240234375 1.0670507812499999 -1.280021484375 1.0664121093749999 -2.11414453125V0.9373359375c-0.000046875 -0.24257812499999998 -0.111890625 -0.471591796875 -0.303169921875 -0.6207773437499999ZM3.300228515625 13.2741796875c-0.808248046875 0.00003515625 -1.3133906249999998 -0.874892578125 -0.9092988281250001 -1.574876953125 0.40409179687499996 -0.699978515625 1.414400390625 -0.700025390625 1.818556640625 -0.00008203125 0.09216796875 0.15962109375 0.140689453125 0.340693359375 0.140689453125 0.5250117187500001 0 0.5798496093750001 -0.47009765625 1.04991796875 -1.049947265625 1.049947265625Zm8.3995546875 -2.099888671875c-0.808248046875 0.00003515625 -1.313396484375 -0.874892578125 -0.9093046875 -1.574876953125 0.40409179687499996 -0.699978515625 1.414400390625 -0.700025390625 1.818556640625 -0.00008203125 0.09216796875 0.15962109375 0.140689453125 0.340693359375 0.140689453125 0.525017578125 0 0.5798496093750001 -0.470091796875 1.04991796875 -1.04994140625 1.04994140625Z" strokeWidth={1} />`,
    `<desc>{"Music 3 Streamline Icon: https://streamlinehq.com"}</desc><path stroke=${colors[2]} strokeLinecap="round" strokeLinejoin="round" d="M4.699999999999999 11.35C4.699999999999999 13.5054375 7.033312499999999 14.8525625 8.9 13.774875C9.7663125 13.274687499999999 10.3 12.350312500000001 10.3 11.35C10.3 9.1945625 7.966687500000001 7.8474375 6.1 8.925125000000001C5.2336875 9.4253125 4.699999999999999 10.3496875 4.699999999999999 11.35" strokeWidth={1} /><path d="M10.3 11.35V0.15" stroke=${colors[2]} strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} />`,
  ];

  useEffect(() => {
    const musicIcon = (animation) => {
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "-0.5 -0.5 16 16");
      svg.setAttribute("fill", "none");
      svg.setAttribute("height", "16");
      svg.setAttribute("width", "16");
      svg.setAttribute("class", `fall-${animation}`);
      svg.innerHTML = icons[animation - 1];

      return svg;
    };

    const handleMouseMove = (event) => {
      let lastX, lastY;
      const icons = document.querySelectorAll(".icon");
      if (icons.length > 0) {
        const lastIcon = icons[icons.length - 1];
        lastX =
          lastIcon.getBoundingClientRect().left + lastIcon.clientWidth / 2;
        lastY =
          lastIcon.getBoundingClientRect().top + lastIcon.clientHeight / 2;
      }

      const glow = document.createElement("div");
      glow.className = "glow"; // Add a class for styling
      glow.style.position = "absolute";
      glow.style.left = `${event.clientX}px`;
      glow.style.top = `${event.clientY}px`;

      document.body.appendChild(glow);

      setTimeout(() => {
        glow.remove();
      }, glowIntervalTime);

      if (
        icons.length === 0 ||
        CalcDistance(lastX, lastY, event.clientX, event.clientY) >
          distanceThreshold
      ) {
        const newIcon = musicIcon(animationCount.current + 1);
        newIcon.classList.add("icon");
        newIcon.style.position = "absolute";
        newIcon.style.left = `${event.clientX}px`;
        newIcon.style.top = `${event.clientY}px`;

        document.body.appendChild(newIcon);

        setTimeout(() => {
          newIcon.remove();
        }, iconIntervalTime);

        animationCount.current = (animationCount.current + 1) % 3;
      }
    };

    // Add event listener to track mouse movement
    window.addEventListener("mousemove", handleMouseMove);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []); // Empty dependency array to run once on mount
}

export default CursorAnimation;
