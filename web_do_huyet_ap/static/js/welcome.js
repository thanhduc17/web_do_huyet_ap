function goLogin() {
  window.location.href = "/login";
}

function goSignup() {
  window.location.href = "/signup"; 
}

// Social buttons actions
document.querySelector('.google')?.addEventListener('click', () => {
  window.location.href = "https://accounts.google.com"; // hoặc link OAuth Google
});

document.querySelector('.apple')?.addEventListener('click', () => {
  window.location.href = "https://appleid.apple.com"; // hoặc link OAuth Apple
});

document.querySelector('.facebook')?.addEventListener('click', () => {
  window.location.href = "https://facebook.com"; // hoặc link OAuth Facebook
});
