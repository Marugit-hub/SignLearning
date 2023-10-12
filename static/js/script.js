var webcamStarted = false; 
var currentImageIndex = 0; 
var handImages = document.querySelectorAll(".hand-container-1 img");


function startWebcam() {
    if (!webcamStarted) {
        var startButton = document.getElementById("startButton");
        webcamStarted = true; 

        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/start_webcam", true);
        xhr.send();

        setTimeout(function() {
            window.location.reload();
        }, 2000); 
    }
}

window.onload = function() {
    if (webcamStarted) {
        renderImage(currentImageIndex); 
    }
};

function redirectToFinish() {
    window.location.href = "/finish";

}

function detectNextLabel() {
    var detectedLabel = "{{ current_label }}";
    if (detectedLabel === "unknown" || detectedLabel === "" || detectedLabel === "error") {
        alert("Please detect the current hand sign before moving to the next.");
    } else {
        var xhr = new XMLHttpRequest();
        xhr.open("GET", "/detect_next_label", true);
        xhr.send();

        xhr.onreadystatechange = function() {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                if (xhr.status === 200 && xhr.responseText === "success") {
                    currentImageIndex++;
                    renderImage(currentImageIndex);
                } else if (xhr.status === 200 && xhr.responseText === "failure") {
                    alert("Current hand sign not detected in the webcam.");
                }
                
            }

        };

    }

}

function renderImage(index) {
    var handImages = document.querySelectorAll(".hand-container-1 img.image");

    if (index > 0) {
        handImages[index - 1].style.transform = "translateX(100%)";
        handImages[index - 1].style.opacity = "0";
    }

    if (index < handImages.length) {
        handImages[index].style.transform = "translateX(0)";
        handImages[index].style.opacity = "1";
    } else {
        alert("All images are shown.");
    }
}


