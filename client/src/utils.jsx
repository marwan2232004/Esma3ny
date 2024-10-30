import * as Tone from "tone";
const startAudioContext = async () => {
    await Tone.start();
};

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

export {startAudioContext,playNote};
