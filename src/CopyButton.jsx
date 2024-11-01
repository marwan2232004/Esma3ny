import "./copyAnimation.css";

// eslint-disable-next-line react/prop-types
const CopyButton = ({ text }) => {

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
  };

  return (
    /* From Uiverse.io by Galahhad */
    <button className="copy" onClick={handleCopy}>
      <span>
        <svg
          xmlSpace="preserve"
          style={{ enableBackground: 'new 0 0 512 512' }}
          viewBox="0 0 6.35 6.35"
          y="0"
          x="0"
          height="20"
          width="20"
          version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          className="clipboard"
        >
          <g>
            <path
              fill="currentColor"
              d="M2.43.265c-.3 0-.548.236-.573.53h-.328a.74.74 0 0 0-.735.734v3.822a.74.74 0 0 0 .735.734H4.82a.74.74 0 0 0 .735-.734V1.529a.74.74 0 0 0-.735-.735h-.328a.58.58 0 0 0-.573-.53zm0 .529h1.49c.032 0 .049.017.049.049v.431c0 .032-.017.049-.049.049H2.43c-.032 0-.05-.017-.05-.049V.843c0-.032.018-.05.05-.05zm-.901.53h.328c.026.292.274.528.573.528h1.49a.58.58 0 0 0 .573-.529h.328a.2.2 0 0 1 .206.206v3.822a.2.2 0 0 1-.206.205H1.53a.2.2 0 0 1-.206-.205V1.529a.2.2 0 0 1 .206-.206z"
            ></path>
          </g>
        </svg>
        <svg
          xmlSpace="preserve"
          style={{ enableBackground: 'new 0 0 512 512' }}
          viewBox="0 0 24 24"
          y="0"
          x="0"
          height="18"
          width="18"
          version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          className="checkmark"
        >
          <g>
            <path
              data-original="#000000"
              fill="currentColor"
              d="M20.285 6.708l-11.4 11.4-4.57-4.57 1.415-1.415 3.155 3.155 10-10z"
            ></path>
          </g>
        </svg>
      </span>
    </button>
  );
};

export default CopyButton;