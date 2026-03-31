import React from 'react';
import { cn } from '../lib/utils';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  children?: React.ReactNode;
  className?: string;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
}

export const Button = ({ 
  className, 
  variant = 'primary', 
  size = 'md', 
  isLoading, 
  children, 
  ...props 
}: ButtonProps) => {
  const variants = {
    primary: 'bg-white text-black hover:bg-[var(--accent-cyan)] shadow-[0_0_20px_rgba(255,255,255,0.1)]',
    secondary: 'bg-white/5 text-white border border-white/10 hover:bg-white/10',
    outline: 'bg-transparent border border-[var(--accent-cyan)] text-[var(--accent-cyan)] hover:bg-[var(--accent-cyan)] hover:text-black',
    ghost: 'bg-transparent text-[var(--text-muted)] hover:text-white',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-6 py-2.5 text-sm',
    lg: 'px-8 py-3 text-base',
  };

  return (
    <button
      className={cn(
        'inline-flex items-center justify-center rounded-md font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-wider',
        variants[variant],
        sizes[size],
        className
      )}
      disabled={Boolean(isLoading || props.disabled)}
      {...props}
    >
      {isLoading ? (
        <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
      ) : null}
      {children}
    </button>
  );
};

export const Card = ({ className, children }: { className?: string; children: React.ReactNode }) => (
  <div className={cn('cyber-border rounded-3xl p-8', className)}>
    {children}
  </div>
);

export const Input = ({ label, ...props }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) => (
  <div className="space-y-2">
    <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
      {label}
    </label>
    <input
      className="w-full bg-white/5 border border-white/10 rounded-full px-6 py-3 text-sm focus:outline-none focus:border-[var(--accent-cyan)] transition-colors text-white"
      {...props}
    />
  </div>
);
