import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/common/Layout';
import SimpleTest from './pages/SimpleTest';
import Analytics from './pages/Analytics';
import Chat from './pages/Chat';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<SimpleTest />} />
              <Route path="analytics" element={<Analytics />} />
              <Route path="chat" element={<Chat />} />
              {/* Future routes */}
              {/* <Route path="customers" element={<Customers />} /> */}
              {/* <Route path="documents" element={<Documents />} /> */}
              {/* <Route path="realtime" element={<Realtime />} /> */}
              {/* <Route path="settings" element={<Settings />} /> */}
            </Route>
          </Routes>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;