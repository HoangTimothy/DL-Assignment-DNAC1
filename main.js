document.querySelectorAll('.card').forEach((card) => {
    card.addEventListener('click', () => {
        console.log('Open page:', card.getAttribute('href'));
    });
});
