import { useState, useEffect } from 'react';
import { useWeb3 } from './hooks/useWeb3';
import { Model } from './services/fheService';
import Marketplace from './pages/Marketplace';
import InferencePortal from './pages/InferencePortal';
import LabDashboard from './pages/LabDashboard';
import LabDatasetsWorkspace from './pages/LabDatasetsWorkspace';
import LabModelsWorkspace from './pages/LabModelsWorkspace';
import ProfileWorkspace from './pages/ProfileWorkspace';
import DataSourceWorkspace from './pages/DataSourceWorkspace';
import { Button } from './components/UI';
import { CursorEffect } from './components/CursorEffect';
import { LayeredStack } from './components/LayeredStack';
import { Shield, Database, ShoppingBag, Beaker, Github, Twitter, MessageSquare, ArrowRight, Cpu, Lock } from 'lucide-react';
import { cn } from './lib/utils';
import { motion, AnimatePresence } from 'motion/react';
import { AppRole, ROLE_DEFINITIONS } from './lib/roles';

type Tab = 'marketplace' | 'inference' | 'lab' | 'profile' | 'upload' | 'datasets' | 'models';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab | 'home'>('home');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [selectedDatasetForModels, setSelectedDatasetForModels] = useState<string | null>(null);
  const {
    address,
    role,
    jwt,
    isConnecting,
    connect,
    disconnect,
    connectionError,
    isAuthenticating,
    isRoleSelectionOpen,
    selectRole,
    dismissRoleSelection,
  } = useWeb3();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSelectModel = (model: Model) => {
    setSelectedModel(model);
    setActiveTab('inference');
  };

  useEffect(() => {
    if (role !== 'ai_lab' && ['lab', 'datasets', 'models'].includes(activeTab)) {
      setActiveTab('home');
    }

    if (role !== 'data_source' && ['upload', 'marketplace', 'inference'].includes(activeTab)) {
      setActiveTab('home');
    }
  }, [role, activeTab]);

  const handleConnectWallet = async () => {
    try {
      await connect();
    } catch (error) {
      console.error('Wallet connection failed:', error);
    }
  };

  const handleChooseRole = async (nextRole: AppRole) => {
    try {
      await selectRole(nextRole);
      setActiveTab('profile');
    } catch (error) {
      console.error('Role selection failed:', error);
    }
  };

  const primaryCta = role === 'ai_lab'
    ? { label: 'Open Lab Dashboard', action: () => setActiveTab('lab') }
    : role === 'data_source'
      ? { label: 'Open Upload Workspace', action: () => setActiveTab('upload') }
      : { label: 'Choose Your Role', action: () => void handleConnectWallet() };

  const secondaryCta = role === 'ai_lab'
    ? { label: 'Browse Datasets', action: () => setActiveTab('datasets') }
    : role === 'data_source'
      ? { label: 'Explore AI Labs', action: () => setActiveTab('marketplace') }
      : { label: 'AI Lab Workspace', action: () => setActiveTab('lab') };

  const NavItem = ({ id, label }: { id: Tab | 'home'; label: string }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={cn(
        'px-4 py-1.5 rounded-full transition-all duration-300 text-[11px] font-bold uppercase tracking-widest',
        activeTab === id 
          ? 'bg-white text-black' 
          : 'text-white/60 hover:text-white'
      )}
    >
      {label}
    </button>
  );

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-main)] selection:bg-[var(--accent-cyan)]/30">
      <CursorEffect />

      {/* Floating Navbar */}
      <div className="fixed top-6 left-0 w-full z-[100] flex justify-center px-6">
        <motion.header 
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          className={cn(
            'glass-pill rounded-full px-2 py-2 flex items-center gap-6 transition-all duration-500',
            scrolled ? 'scale-95' : 'scale-100'
          )}
        >
          <div className="flex items-center gap-3 pl-4 pr-6 border-r border-white/10">
            <Shield className="w-5 h-5 text-[var(--accent-cyan)]" />
            <span className="text-sm font-black tracking-tighter">BLINFERENCE</span>
          </div>

          <nav className="flex items-center gap-1">
            <NavItem id="home" label="Home" />
            {role && <NavItem id="profile" label="Profile" />}
            {role === 'data_source' && <NavItem id="marketplace" label="Marketplace" />}
            {role === 'data_source' && <NavItem id="upload" label="Upload" />}
            {role === 'data_source' && <NavItem id="inference" label="Portal" />}
            {role === 'ai_lab' && <NavItem id="lab" label="Lab" />}
            {role === 'ai_lab' && <NavItem id="datasets" label="Datasets" />}
            {role === 'ai_lab' && <NavItem id="models" label="Models" />}
          </nav>

          <div className="flex items-center gap-2 pl-4 pr-2">
            {address ? (
              <button 
                onClick={disconnect}
                className="flex items-center gap-2 px-4 py-1.5 bg-white/5 hover:bg-white/10 rounded-full border border-white/10 transition-all"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--status-success)]" />
                <span className="text-[10px] font-mono">{address.substring(0, 6)}...</span>
                {role && (
                  <span className="rounded-full bg-white/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                    {ROLE_DEFINITIONS[role].badge}
                  </span>
                )}
                {jwt && (
                  <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest text-emerald-300">
                    AUTH
                  </span>
                )}
              </button>
            ) : (
              <button 
                onClick={handleConnectWallet}
                className="bg-white text-black px-5 py-1.5 rounded-full text-[11px] font-bold uppercase tracking-widest hover:bg-[var(--accent-cyan)] transition-colors"
              >
                {isConnecting ? 'Connecting...' : 'Connect'}
              </button>
            )}
          </div>
        </motion.header>
      </div>

      {connectionError && (
        <div className="fixed top-24 left-1/2 z-[90] w-full max-w-xl -translate-x-1/2 px-6">
          <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200 backdrop-blur">
            {connectionError}
          </div>
        </div>
      )}

      <AnimatePresence>
        {isRoleSelectionOpen && address && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 px-6 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, y: 24, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 24, scale: 0.96 }}
              className="w-full max-w-5xl rounded-[2rem] border border-white/10 bg-[var(--bg-secondary)]/95 p-8 shadow-2xl"
            >
              <div className="mb-8 flex items-start justify-between gap-6">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.35em] text-[var(--accent-cyan)]">
                    Phase 1 Onboarding
                  </div>
                  <h2 className="mt-3 text-4xl font-black uppercase tracking-tight">
                    Choose Your Role
                  </h2>
                  <p className="mt-3 max-w-2xl text-sm text-[var(--text-muted)]">
                    This choice is mapped to your connected wallet in the app shell for Wave 1. Data Sources upload training datasets. AI Labs activate on-chain, browse encrypted datasets, and manage supply-side workflows.
                  </p>
                  <p className="mt-3 text-xs font-mono text-white/40">
                    Wallet: {address}
                  </p>
                </div>
                <button
                  onClick={dismissRoleSelection}
                  disabled={isAuthenticating}
                  className="rounded-full border border-white/10 px-3 py-2 text-xs uppercase tracking-widest text-white/50 transition-colors hover:text-white"
                >
                  Later
                </button>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                {(Object.entries(ROLE_DEFINITIONS) as [AppRole, typeof ROLE_DEFINITIONS[AppRole]][]).map(
                  ([key, roleDef]) => (
                    <button
                      key={key}
                      onClick={() => void handleChooseRole(key)}
                      disabled={isAuthenticating}
                      className="group rounded-[1.75rem] border border-white/10 bg-white/[0.02] p-8 text-left transition-all hover:border-[var(--accent-cyan)]/40 hover:bg-white/[0.04] disabled:cursor-wait disabled:opacity-60"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="text-[10px] font-bold uppercase tracking-[0.35em] text-[var(--accent-cyan)]">
                            {roleDef.badge}
                          </div>
                          <h3 className="mt-3 text-3xl font-black uppercase tracking-tight">
                            {roleDef.label}
                          </h3>
                          <p className="mt-3 text-sm text-[var(--text-main)]">
                            {roleDef.tagline}
                          </p>
                        </div>
                        <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-[var(--accent-cyan)] transition-transform group-hover:scale-110">
                          {key === 'ai_lab' ? <Beaker className="h-7 w-7" /> : <Database className="h-7 w-7" />}
                        </div>
                      </div>

                      <p className="mt-6 max-w-xl text-sm leading-relaxed text-[var(--text-muted)]">
                        {roleDef.summary}
                      </p>

                      <div className="mt-6 space-y-2">
                        {roleDef.capabilities.map((capability) => (
                          <div
                            key={capability}
                            className="flex items-center gap-3 text-xs uppercase tracking-widest text-white/55"
                          >
                            <div className="h-1.5 w-1.5 rounded-full bg-[var(--accent-cyan)]" />
                            <span>{capability}</span>
                          </div>
                        ))}
                      </div>

                      <div className="mt-8 inline-flex items-center text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                        {isAuthenticating ? 'Requesting Wallet Signature' : `Continue as ${roleDef.label}`}
                        <ArrowRight className="ml-2 h-4 w-4" />
                      </div>
                    </button>
                  ),
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="pt-32 pb-24">
        <AnimatePresence mode="wait">
          {activeTab === 'home' && (
            <motion.div 
              key="home"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-7xl mx-auto px-6"
            >
              {/* Hero Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center min-h-[70vh]">
                <div className="space-y-8">
                  <motion.div
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                  >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--accent-cyan)]/10 border border-[var(--accent-cyan)]/20 text-[var(--accent-cyan)] text-[10px] font-bold uppercase tracking-[0.2em] mb-6">
                      <div className="w-1 h-1 rounded-full bg-[var(--accent-cyan)] animate-ping" />
                      {role ? `${ROLE_DEFINITIONS[role].label} Workspace` : 'FHE-Powered Blind Inference'}
                    </div>
                    <h1 className="text-7xl font-black tracking-tighter leading-[0.9] uppercase">
                      Private AI <br />
                      For <span className="neon-text">{role === 'ai_lab' ? 'Model Builders' : role === 'data_source' ? 'Data Sources' : 'Two Clear Roles'}</span> <br />
                      On Fhenix
                    </h1>
                  </motion.div>
                  
                  <motion.p 
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="text-xl text-[var(--text-muted)] max-w-lg leading-relaxed"
                  >
                    {role
                      ? ROLE_DEFINITIONS[role].summary
                      : 'Connect a wallet, choose whether you are a Data Source or an AI Lab, and enter a role-aware confidential AI workflow without exposing plaintext data.'}
                  </motion.p>

                  <motion.div 
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="flex gap-4"
                  >
                    <Button size="lg" onClick={primaryCta.action}>
                      {primaryCta.label} <ArrowRight className="ml-2 w-4 h-4" />
                    </Button>
                    <Button variant="secondary" size="lg" onClick={secondaryCta.action}>
                      {secondaryCta.label}
                    </Button>
                  </motion.div>
                </div>

                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.5 }}
                >
                  <LayeredStack />
                </motion.div>
              </div>

              {/* Feature Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-32">
                {[
                  { title: 'Data Sources', desc: 'Upload protected inputs and request confidential compute without exposing raw records to model providers.', icon: Database },
                  { title: 'AI Labs', desc: 'Register encrypted models, price inference, and operate the supply side of the confidential AI marketplace.', icon: Beaker },
                  { title: 'Shared FHE Rail', desc: 'Both roles use the same Fhenix-powered privacy rail for encrypted inputs, pricing, execution, and selective decryption.', icon: Shield },
                ].map((f, i) => (
                  <motion.div
                    key={i}
                    initial={{ y: 50, opacity: 0 }}
                    whileInView={{ y: 0, opacity: 1 }}
                    transition={{ delay: i * 0.1 }}
                    viewport={{ once: true }}
                    className="p-8 rounded-3xl bg-white/[0.02] border border-white/5 hover:border-[var(--accent-cyan)]/30 transition-all group"
                  >
                    <f.icon className="w-10 h-10 text-[var(--accent-cyan)] mb-6 group-hover:scale-110 transition-transform" />
                    <h3 className="text-xl font-bold mb-3">{f.title}</h3>
                    <p className="text-sm text-[var(--text-muted)] leading-relaxed">{f.desc}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'marketplace' && (
            <div className="max-w-7xl mx-auto px-6">
              <Marketplace onSelectModel={handleSelectModel} />
            </div>
          )}

          {activeTab === 'profile' && (
            <div className="max-w-7xl mx-auto px-6">
              <ProfileWorkspace />
            </div>
          )}

          {activeTab === 'upload' && (
            <div className="max-w-7xl mx-auto px-6">
              <DataSourceWorkspace />
            </div>
          )}

          {activeTab === 'inference' && (
            <div className="max-w-7xl mx-auto px-6">
              {selectedModel ? (
                <InferencePortal model={selectedModel} />
              ) : (
                <div className="flex flex-col items-center justify-center py-24 space-y-6 text-center">
                  <ShoppingBag className="w-16 h-16 text-white/10" />
                  <h2 className="text-2xl font-bold">No Model Selected</h2>
                  <Button onClick={() => setActiveTab('marketplace')}>Go to Marketplace</Button>
                </div>
              )}
            </div>
          )}

          {activeTab === 'lab' && (
            <div className="max-w-7xl mx-auto px-6">
              <LabDashboard />
            </div>
          )}

          {activeTab === 'datasets' && (
            <div className="max-w-7xl mx-auto px-6">
              <LabDatasetsWorkspace
                onOpenModels={(datasetId) => {
                  setSelectedDatasetForModels(datasetId);
                  setActiveTab('models');
                }}
              />
            </div>
          )}

          {activeTab === 'models' && (
            <div className="max-w-7xl mx-auto px-6">
              <LabModelsWorkspace initialDatasetId={selectedDatasetForModels} />
            </div>
          )}
        </AnimatePresence>
      </main>

      {/* High-Fidelity Footer */}
      <footer className="relative mt-32 border-t border-white/5 bg-[var(--bg-secondary)] overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[300px] bg-[var(--accent-cyan)]/5 blur-[120px] rounded-full" />
        
        <div className="max-w-7xl mx-auto px-6 py-24 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-24 mb-24">
            <div className="space-y-8">
              <div className="flex items-center gap-3">
                <Shield className="w-8 h-8 text-[var(--accent-cyan)]" />
                <span className="text-2xl font-black tracking-tighter">BLINFERENCE</span>
              </div>
              <p className="text-xl text-[var(--text-muted)] max-w-md">
                Building the foundational layer for private, secure, and decentralized artificial intelligence.
              </p>
              <div className="flex gap-4">
                {[Github, Twitter, MessageSquare].map((Icon, i) => (
                  <a key={i} href="#" className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center hover:bg-[var(--accent-cyan)] hover:text-black transition-all">
                    <Icon className="w-5 h-5" />
                  </a>
                ))}
              </div>
            </div>

            <div className="space-y-8">
              <h3 className="text-lg font-bold">Sign up for our newsletter</h3>
              <div className="flex gap-2">
                <input 
                  type="email" 
                  placeholder="Your Email" 
                  className="flex-1 bg-white/5 border border-white/10 rounded-full px-6 py-3 text-sm focus:outline-none focus:border-[var(--accent-cyan)] transition-colors"
                />
                <button className="bg-white text-black px-8 py-3 rounded-full text-sm font-bold hover:bg-[var(--accent-cyan)] transition-colors">
                  Subscribe
                </button>
              </div>
              <p className="text-xs text-[var(--text-muted)]">
                Stay updated with the latest FHE research and protocol updates.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-12 pt-12 border-t border-white/5">
            {[
              { title: 'Develop', links: ['Testnet', 'Faucet', 'Docs', 'Storage Scan'] },
              { title: 'Learn', links: ['Blog', 'AMAs', 'FAQs', 'Whitepaper'] },
              { title: 'Ecosystem', links: ['Accelerator', 'Press', 'Contact Us', 'Brand Kit'] },
              { title: 'Legal', links: ['Privacy Policy', 'Terms of Service', 'Cookie Policy'] },
            ].map((col, i) => (
              <div key={i} className="space-y-4">
                <h4 className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/40">{col.title}</h4>
                <ul className="space-y-2">
                  {col.links.map((link, j) => (
                    <li key={`${col.title}-${link || 'link'}-${j}`}>
                      <a href="#" className="text-sm text-[var(--text-muted)] hover:text-white transition-colors">{link}</a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          <div className="mt-24 pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.3em]">
              &copy; 2026 BLINFERENCE PROTOCOL // ZERO KNOWLEDGE AI
            </div>
            <div className="flex items-center gap-2 text-[10px] font-bold text-white/20 uppercase tracking-[0.3em]">
              <div className="w-1 h-1 rounded-full bg-[var(--status-success)]" />
              All Systems Operational
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
