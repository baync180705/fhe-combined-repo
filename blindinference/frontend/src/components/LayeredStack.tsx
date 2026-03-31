import { motion } from 'motion/react';
import { Shield, Cpu, Lock, Database } from 'lucide-react';

export const LayeredStack = () => {
  return (
    <div className="relative w-full h-[500px] flex items-center justify-center layered-stack">
      {/* Background Grid */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none" />
      
      {/* Stack Layers */}
      <div className="relative w-64 h-64">
        {/* Top Layer - Application */}
        <motion.div
          animate={{ y: [-20, -40, -20], rotateX: 45, rotateZ: -45 }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          className="absolute inset-0 bg-[var(--accent-cyan)]/20 border border-[var(--accent-cyan)] rounded-2xl flex items-center justify-center backdrop-blur-md shadow-[0_0_50px_rgba(0,242,255,0.2)]"
          style={{ transform: 'rotateX(45deg) rotateZ(-45deg) translateZ(100px)' }}
        >
          <Shield className="w-12 h-12 text-[var(--accent-cyan)]" />
          <div className="absolute -top-8 text-[10px] font-bold uppercase tracking-[0.3em] text-[var(--accent-cyan)]">Application Layer</div>
        </motion.div>

        {/* Middle Layer - FHE Engine */}
        <motion.div
          animate={{ y: [0, -10, 0], rotateX: 45, rotateZ: -45 }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
          className="absolute inset-0 bg-white/5 border border-white/20 rounded-2xl flex items-center justify-center backdrop-blur-sm"
          style={{ transform: 'rotateX(45deg) rotateZ(-45deg) translateZ(50px)' }}
        >
          <Cpu className="w-12 h-12 text-white/50" />
          <div className="absolute -right-24 text-[10px] font-bold uppercase tracking-[0.3em] text-white/30">FHE Compute Engine</div>
        </motion.div>

        {/* Bottom Layer - Data Privacy */}
        <motion.div
          animate={{ y: [20, 10, 20], rotateX: 45, rotateZ: -45 }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          className="absolute inset-0 bg-white/5 border border-white/10 rounded-2xl flex items-center justify-center"
          style={{ transform: 'rotateX(45deg) rotateZ(-45deg)' }}
        >
          <Lock className="w-12 h-12 text-white/20" />
          <div className="absolute -bottom-8 text-[10px] font-bold uppercase tracking-[0.3em] text-white/20">Encrypted Data Layer</div>
        </motion.div>

        {/* Connecting Lines (Simplified) */}
        <div className="absolute inset-0 border-l border-dashed border-white/10 h-[200px] -translate-x-12 translate-y-12" />
      </div>
    </div>
  );
};
