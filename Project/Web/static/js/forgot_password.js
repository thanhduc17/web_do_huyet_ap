// ⚡ Firebase config (cần thay bằng project của bạn)
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT.firebaseio.com",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_MSG_ID",
  appId: "YOUR_APP_ID"
};

firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const database = firebase.database();

function showMessage(msg, color = "red") {
  const messageEl = document.getElementById("message");
  messageEl.textContent = msg;
  messageEl.style.color = color;
}

function sendPasswordReset() {
  const email = document.getElementById("email").value.trim();

  if (!email) {
    showMessage("Please enter your email.");
    return;
  }

  // Kiểm tra email có tồn tại trong Realtime Database
  database.ref("users").orderByChild("email").equalTo(email).once("value")
    .then(snapshot => {
      if (snapshot.exists()) {
        // Nếu có, gửi mail reset password
        auth.sendPasswordResetEmail(email)
          .then(() => {
            showMessage("Password reset email sent!", "green");
          })
          .catch(error => {
            showMessage("Error: " + error.message);
          });
      } else {
        showMessage("Email not registered.");
      }
    })
    .catch(error => {
      showMessage("Error: " + error.message);
    });
}
