import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // Set to '/' if deploying to a root domain (e.g. a user/org Pages site).
  // Set to '/your-repo-name/' for a project Pages site.
  base: '/Particles1/',
  plugins: [react()],
});
