import { useEffect, useState, type ReactNode } from 'react';

type RuntimeErrorOverlayProps = {
  children: ReactNode;
};

export function RuntimeErrorOverlay({ children }: RuntimeErrorOverlayProps) {
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const onError = (event: ErrorEvent) => {
      setMessage(event.error?.message ?? event.message ?? 'Unknown runtime error');
    };

    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason;
      if (reason instanceof Error) {
        setMessage(reason.message);
        return;
      }
      if (typeof reason === 'string') {
        setMessage(reason);
        return;
      }
      setMessage('Unhandled promise rejection');
    };

    window.addEventListener('error', onError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);

    return () => {
      window.removeEventListener('error', onError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
    };
  }, []);

  if (message) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] px-6 py-20 text-[var(--text-main)]">
        <div className="mx-auto max-w-3xl rounded-3xl border border-rose-500/30 bg-rose-500/10 p-8">
          <h1 className="text-2xl font-bold text-rose-300">Frontend Error</h1>
          <p className="mt-3 text-sm text-rose-100/80">
            The app hit a runtime error instead of rendering normally.
          </p>
          <pre className="mt-6 overflow-x-auto rounded-2xl border border-white/10 bg-black/30 p-4 text-xs text-rose-100/90">
            {message}
          </pre>
          <p className="mt-4 text-xs text-[var(--text-muted)]">
            Check the browser console too. If you send me the message above, I can pinpoint the next fix.
          </p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
