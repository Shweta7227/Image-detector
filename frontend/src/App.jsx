import { useState } from 'react'

function App() {
  const [file, setFile] = useState(null);

  return (
    <div className="container my-5">
      <h1 className="text-center mb-4">Deepfake Image Detector</h1>

      <div className="card mx-auto" style={{ maxWidth: '500px' }}>
        <div className="card-body">
          <h5 className="card-title">Upload Image</h5>

          <div className="mb-3">
            <input 
              type="file" 
              className="form-control" 
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
          </div>

          {file && (
            <div className="text-center">
              <img 
                src={URL.createObjectURL(file)} 
                alt="Preview" 
                className="img-fluid rounded mb-3" 
                style={{ maxHeight: '300px' }}
              />
              <button className="btn btn-primary w-100">
                Detect Deepfake
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App