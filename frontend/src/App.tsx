/**
 * Root application component with routing.
 *
 * Defines all routes and wraps in AuthProvider.
 * Unauthenticated users are redirected to /login.
 */

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { PreferencesProvider } from "./context/PreferencesContext";
import Layout from "./components/Layout";
import LoginPage from "./pages/LoginPage";
import UploadPage from "./pages/UploadPage";
import ProcessingPage from "./pages/ProcessingPage";
import TranscriptPage from "./pages/TranscriptPage";
import ProtocolPage from "./pages/ProtocolPage";
import MeetingsPage from "./pages/MeetingsPage";
import SpeakersPage from "./pages/SpeakersPage";
import SystemPage from "./pages/SystemPage";
import AuditPage from "./pages/AuditPage";
import UsersPage from "./pages/UsersPage";
import { Loader2 } from "lucide-react";

/** Route guard that redirects to /login if not authenticated. */
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 size={32} className="animate-spin text-epam-red" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function AppRoutes() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-epam-gray-50">
        <div className="text-center">
          <Loader2 size={32} className="animate-spin text-epam-red mx-auto mb-3" />
          <p className="text-sm text-epam-gray-500">Connecting to VERITAS...</p>
        </div>
      </div>
    );
  }

  return (
    <Routes>
      <Route
        path="/login"
        element={
          isAuthenticated ? <Navigate to="/upload" replace /> : <LoginPage />
        }
      />

      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/upload" replace />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="processing/:jobId" element={<ProcessingPage />} />
        <Route path="transcript/:jobId" element={<TranscriptPage />} />
        <Route path="protocol/:jobId" element={<ProtocolPage />} />
        <Route path="meetings" element={<MeetingsPage />} />
        <Route path="speakers" element={<SpeakersPage />} />
        <Route path="users" element={<UsersPage />} />
        <Route path="system" element={<SystemPage />} />
        <Route path="audit" element={<AuditPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <PreferencesProvider>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </PreferencesProvider>
    </BrowserRouter>
  );
}
