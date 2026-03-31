import { FormEvent, useEffect, useState } from 'react';
import { Card, Button, Input } from '../components/UI';
import { useWeb3 } from '../hooks/useWeb3';
import { ROLE_DEFINITIONS } from '../lib/roles';
import { getUserProfile, saveUserProfile } from '../services/profileService';
import { fetchUserProfileFromIpfs, isIpfsUri, uploadUserProfileToIpfs } from '../services/ipfsProfileService';
import { BadgeInfo, Building2, FileText, Loader2, Save, UserRound, ShieldCheck } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';

function defaultProfileUri(address: string) {
  const normalized = address.toLowerCase();
  return `blindference://labs/${normalized}`;
}

function truncateWallet(address: string) {
  return address.length > 25 ? `${address.slice(0, 25)}...` : address;
}

export default function ProfileWorkspace() {
  const { address, role, jwt } = useWeb3();
  const [displayName, setDisplayName] = useState('');
  const [organization, setOrganization] = useState('');
  const [bio, setBio] = useState('');
  const [profileUri, setProfileUri] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const roleDef = role ? ROLE_DEFINITIONS[role] : null;

  useEffect(() => {
    if (!address || !role || !jwt) {
      return;
    }

    let isActive = true;

    async function loadProfile() {
      setIsLoading(true);
      setError(null);
      try {
        const profile = await getUserProfile(address, jwt);
        if (!isActive) {
          return;
        }

        setDisplayName(profile.display_name ?? '');
        setOrganization(profile.organization ?? '');
        setBio(profile.bio ?? '');
        const canonicalProfileUri = isIpfsUri(profile.profile_uri) ? profile.profile_uri : '';
        setProfileUri(canonicalProfileUri);

        if (canonicalProfileUri) {
          try {
            const ipfsProfile = await fetchUserProfileFromIpfs(canonicalProfileUri);
            if (!isActive) {
              return;
            }

            setDisplayName(ipfsProfile.display_name ?? '');
            setOrganization(ipfsProfile.organization ?? '');
            setBio(ipfsProfile.bio ?? '');
          } catch (ipfsError) {
            if (!isActive) {
              return;
            }

            console.error('Failed to fetch profile from IPFS:', ipfsError);
          }
        }
      } catch (loadError) {
        if (!isActive) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : 'Failed to load profile');
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadProfile();

    return () => {
      isActive = false;
    };
  }, [address, role, jwt]);

  const handleSave = async (event: FormEvent) => {
    event.preventDefault();
    if (!address || !role || !jwt) {
      setError('Connect and authenticate your wallet before editing your profile.');
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const timestamp = new Date().toISOString();
      const nextProfileUri = await uploadUserProfileToIpfs({
        version: 1,
        address,
        role,
        display_name: displayName.trim() || null,
        organization: organization.trim() || null,
        bio: bio.trim() || null,
        created_at: timestamp,
        updated_at: timestamp,
      });

      const saved = await saveUserProfile(address, jwt, {
        display_name: '',
        organization: '',
        bio: '',
        profile_uri: nextProfileUri,
      });

      setDisplayName(displayName.trim());
      setOrganization(organization.trim());
      setBio(bio.trim());
      setProfileUri(saved.profile_uri ?? nextProfileUri);
      setSuccess('Profile saved to IPFS successfully.');
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Failed to save profile');
    } finally {
      setIsSaving(false);
    }
  };

  if (!address || !role || !jwt) {
    return (
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center py-24 text-center">
        <div className="mb-6 rounded-full border border-white/10 bg-white/5 p-5">
          <UserRound className="h-10 w-10 text-[var(--accent-cyan)]" />
        </div>
        <h1 className="text-3xl font-black uppercase tracking-tight">Connect To Manage Profile</h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
          Your wallet-backed profile stores the app-layer metadata for your role. Connect and authenticate first to edit your Data Source or AI Lab identity.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center gap-4">
        <div className="rounded-xl bg-[var(--accent-cyan)]/10 p-3">
          <UserRound className="h-8 w-8 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight neon-text">Profile Workspace</h1>
          <p className="text-[var(--text-muted)]">
            Manage the wallet-bound metadata for your {roleDef?.label.toLowerCase()} role.
          </p>
        </div>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <Card>
            <form onSubmit={handleSave} className="space-y-6">
              <div className="grid gap-6 md:grid-cols-2">
                <div className="md:col-span-2">
                  <Input
                    label={role === 'ai_lab' ? 'Lab Display Name' : 'Display Name'}
                    type="text"
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
                    placeholder={role === 'ai_lab' ? 'Fhenix Research Lab' : 'St. Mary Data Unit'}
                    required
                  />
                </div>

                <Input
                  label="Organization"
                  type="text"
                  value={organization}
                  onChange={(event) => setOrganization(event.target.value)}
                  placeholder={role === 'ai_lab' ? 'Blindference Labs' : 'St. Mary Hospital'}
                />

                <div className="space-y-2">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                    Wallet
                  </label>
                  <div className="rounded-full border border-white/10 bg-white/5 px-6 py-3 text-sm text-white/75">
                    {truncateWallet(address)}
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                  Bio
                </label>
                <textarea
                  className="min-h-36 w-full rounded-3xl border border-white/10 bg-white/5 px-6 py-4 text-sm text-white focus:border-[var(--accent-cyan)] focus:outline-none"
                  value={bio}
                  onChange={(event) => setBio(event.target.value)}
                  placeholder={
                    role === 'ai_lab'
                      ? 'Describe your models, domain expertise, and why Data Sources should trust your lab.'
                      : 'Describe your data domain, privacy expectations, and what you want to use blind inference for.'
                  }
                />
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                  Canonical IPFS Profile URI
                </label>
                <div className="rounded-3xl border border-white/10 bg-white/5 px-6 py-4 text-sm text-white/75 break-all">
                  {profileUri || 'Profile not published yet. Saving will pin the latest JSON to IPFS.'}
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-white/5 pt-4">
                <div className="text-xs text-[var(--text-muted)]">
                  {role === 'ai_lab'
                    ? 'Saving publishes your profile JSON to IPFS and the resulting URI is what the AI Lab contract call will use.'
                    : 'Saving publishes your wallet profile JSON to IPFS and keeps only the resulting URI in app storage.'}
                </div>
                <Button type="submit" isLoading={isSaving} disabled={isLoading}>
                  <Save className="mr-2 h-4 w-4" />
                  Save Profile
                </Button>
              </div>
            </form>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <div className="flex items-center gap-3">
              <BadgeInfo className="h-5 w-5 text-[var(--accent-cyan)]" />
              <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
                Role Summary
              </h3>
            </div>
            <p className="mt-4 text-sm text-white">{roleDef?.tagline}</p>
            <p className="mt-4 text-xs leading-relaxed text-[var(--text-muted)]">{roleDef?.summary}</p>
          </Card>

          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <div className="flex items-center gap-3">
              <Building2 className="h-5 w-5 text-[var(--accent-cyan)]" />
              <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
                Completion Hints
              </h3>
            </div>
            <div className="mt-4 space-y-3 text-xs leading-relaxed text-[var(--text-muted)]">
              <p>
                <span className="font-bold text-white">Display Name:</span> what the interface should show for your wallet.
              </p>
              <p>
                <span className="font-bold text-white">Organization:</span> the institution or lab operating this wallet.
              </p>
              <p>
                <span className="font-bold text-white">Bio:</span> enough context for trust, discovery, and reviewer clarity.
              </p>
              <p>
                <span className="font-bold text-white">IPFS URI:</span> generated automatically when you save, then reused anywhere Blindference needs your canonical profile reference.
              </p>
            </div>
          </Card>

          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-white/10 bg-white/[0.03] p-4 text-sm text-white/70"
              >
                <Loader2 className="mr-2 inline h-4 w-4 animate-spin" />
                Loading profile...
              </motion.div>
            )}
            {success && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-200"
              >
                <ShieldCheck className="mr-2 inline h-4 w-4" />
                {success}
              </motion.div>
            )}
            {error && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200"
              >
                <FileText className="mr-2 inline h-4 w-4" />
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
