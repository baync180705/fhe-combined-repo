import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import { RuntimeErrorOverlay } from './components/RuntimeErrorOverlay.tsx';
import { Web3Provider } from './hooks/useWeb3.ts';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RuntimeErrorOverlay>
      <Web3Provider>
        <App />
      </Web3Provider>
    </RuntimeErrorOverlay>
  </StrictMode>,
);
