import { useCallback, useEffect, useState } from 'react';
import { CalendarView } from './components/CalendarView';
import { CallStage } from './components/CallStage';
import { CrmView } from './components/CrmView';
import { InsightsView } from './components/InsightsView';
import { LeadSidePanel } from './components/LeadSidePanel';
import { Navbar } from './components/Navbar';
import { TranscriptPanel } from './components/TranscriptPanel';
import { useVoiceCall } from './hooks/useVoiceCall';
import { fetchCalendar, fetchDashboard } from './lib/api';
import type { CalendarData, DashboardData, TabId } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<TabId>('voice');
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [dashboardErr, setDashboardErr] = useState<string | null>(null);
  const [calendar, setCalendar] = useState<CalendarData | null>(null);
  const [calendarErr, setCalendarErr] = useState<string | null>(null);

  const loadDashboard = useCallback(async () => {
    try {
      setDashboardErr(null);
      const data = await fetchDashboard();
      setDashboard(data);
    } catch (err) {
      setDashboardErr(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const loadCalendar = useCallback(async () => {
    try {
      setCalendarErr(null);
      const data = await fetchCalendar();
      setCalendar(data);
    } catch (err) {
      setCalendarErr(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const voice = useVoiceCall(loadDashboard);

  // Initial badge counts on boot
  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  // Refresh data when switching to Calendar, CRM, or Insights
  const handleSelectTab = useCallback(
    (tab: TabId) => {
      setActiveTab(tab);
      if (tab === 'calendar') void loadCalendar();
      if (tab === 'crm' || tab === 'insights') void loadDashboard();
    },
    [loadCalendar, loadDashboard]
  );

  // Keyboard shortcuts: Space starts/ends call, M toggles mute
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (document.activeElement as HTMLElement | null)?.tagName;
      if (
        (tag && ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) ||
        e.metaKey ||
        e.ctrlKey ||
        e.altKey
      ) {
        return;
      }
      if (e.code === 'Space' && activeTab === 'voice') {
        e.preventDefault();
        voice.toggleCall();
      } else if (e.key === 'm' || e.key === 'M') {
        voice.toggleMute();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [activeTab, voice.toggleCall, voice.toggleMute]);

  return (
    <>
      <div className="aurora" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>

      <Navbar
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        turnCount={voice.turnCount}
        demoCount={calendar?.appointments.length ?? dashboard?.appointments.length ?? 0}
        leadCount={dashboard?.leads.length ?? 0}
        summaryCount={dashboard?.summaries.length ?? 0}
        language={voice.language}
        onLanguageChange={voice.setLanguage}
        onNewSession={voice.newSession}
        callState={voice.callState}
        inCall={voice.inCall}
        wsStatus={voice.wsStatus}
      />

      <main>
        {/* Keep Call view mounted so an active call never drops when inspecting Calendar/CRM */}
        <section
          id="view-voice"
          className={`call-grid${activeTab === 'voice' ? '' : ' hidden'}`}
        >
          <div className="col-main">
            <CallStage
              sessionId={voice.sessionId}
              elapsedSec={voice.elapsedSec}
              callState={voice.callState}
              caption={voice.caption}
              inCall={voice.inCall}
              muted={voice.muted}
              bargeIn={voice.bargeIn}
              micHelp={voice.micHelp}
              meterBars={voice.meterBars}
              metrics={voice.metrics}
              onToggleCall={voice.toggleCall}
              onToggleMute={voice.toggleMute}
              onBargeInChange={voice.setBargeIn}
            />
            <TranscriptPanel
              entries={voice.entries}
              turnCount={voice.turnCount}
              inCall={voice.inCall}
              onSendText={voice.sendText}
            />
          </div>

          <LeadSidePanel
            lead={voice.lead}
            ragSources={voice.ragSources}
            summary={voice.summary}
            summarizing={voice.summarizing}
            onSummarize={voice.generateSummary}
          />
        </section>

        {activeTab === 'calendar' && (
          <CalendarView data={calendar} error={calendarErr} onRefresh={loadCalendar} />
        )}

        {activeTab === 'crm' && (
          <CrmView data={dashboard} error={dashboardErr} onRefresh={loadDashboard} />
        )}

        {activeTab === 'insights' && (
          <InsightsView data={dashboard} error={dashboardErr} onRefresh={loadDashboard} />
        )}
      </main>
    </>
  );
}
