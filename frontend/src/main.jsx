import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import 'bootstrap/dist/css/bootstrap.min.css';
// Optional: if you need Bootstrap's JS features (tooltips, modals, dropdowns, etc.)
// import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
