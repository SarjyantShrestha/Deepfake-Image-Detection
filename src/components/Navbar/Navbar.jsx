import "./Navbar.css";
import githublogo from "../../Assets/githublogo.png";
import Uploadarea from "../Uploadarea/Uploadarea";

function Navbar() {
  return (
    <div className="header">
      <h1>DeepFake Image Detector</h1>
      <div className="nav">
        {/* <p id="upload">Upload file</p> */}
        <div className="image">
          <a
            id="a"
            href="https://github.com/SarjyantShrestha/Deepfake-detection-project"
            target="_self"
          >
            <img id="gitimg" src={githublogo} alt="" />
          </a>
        </div>
      </div>
      <hr className="line" />
      <Uploadarea />
    </div>
  );
}

export default Navbar;
