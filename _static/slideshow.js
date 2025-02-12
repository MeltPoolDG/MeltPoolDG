document.addEventListener("DOMContentLoaded", function() {
    let currentIndex = 0;
    const videos = document.querySelectorAll('.youtube-slideshow iframe');

    // If videos are found, run the slideshow
    if (videos.length > 0) {
        function showNextVideo() {
            // Hide the current video
            videos[currentIndex].style.display = 'none';

            // Move to the next video
            currentIndex = (currentIndex + 1) % videos.length;

            // Show the next video
            videos[currentIndex].style.display = 'block';
        }

        function showPrevVideo() {
            // Hide the current video
            videos[currentIndex].style.display = 'none';

            // Move to the previous video
            currentIndex = (currentIndex - 1 + videos.length) % videos.length;

            // Show the previous video
            videos[currentIndex].style.display = 'block';
        }

        // Create buttons for manual navigation with arrow symbols
        const nextButton = document.createElement('button');
        nextButton.textContent = '→';  // Right arrow
        nextButton.classList.add('slideshow-button', 'next-button');
        nextButton.addEventListener('click', showNextVideo);

        const prevButton = document.createElement('button');
        prevButton.textContent = '←';  // Left arrow
        prevButton.classList.add('slideshow-button', 'prev-button');
        prevButton.addEventListener('click', showPrevVideo);

        // Append buttons to the container
        const slideshowContainer = document.querySelector('.youtube-slideshow');
        if (slideshowContainer) {
            slideshowContainer.appendChild(prevButton);
            slideshowContainer.appendChild(nextButton);
        }

        // Start the slideshow by showing the next video every 5 seconds
        setInterval(showNextVideo, 5000);

        // Display the first video
        videos[currentIndex].style.display = 'block';
    }
});
