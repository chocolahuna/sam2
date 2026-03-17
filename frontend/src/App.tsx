import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [loading, setLoading] = useState(false)
  const [projectId, setProjectId] = useState<string | null>(null)
  const [projectData, setProjectData] = useState<any>(null)
  const [bgBrightness, setBgBrightness] = useState(30) // 0-100%
  const [selectedPartIndex, setSelectedPartIndex] = useState<number | null>(null)
  const [activeTool, setActiveTool] = useState<"none" | "point" | "lasso">("none")
  const [currentPoints, setCurrentPoints] = useState<{x: number, y: number, label: number, dispX: number, dispY: number}[]>([])
  
  // Lasso State
  const [lassoPoints, setLassoPoints] = useState<{x: number, y: number, dispX: number, dispY: number}[]>([])
  const [isDrawingLasso, setIsDrawingLasso] = useState(false)
  const [editingPartId, setEditingPartId] = useState<string | null>(null)
  const [projectsList, setProjectsList] = useState<any[]>([])

  const fetchProjects = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/projects')
      const data = await response.json()
      setProjectsList(data)
    } catch (error) {
      console.error('Error fetching projects:', error)
    }
  }

  const loadProject = async (id: string) => {
    setLoading(true)
    try {
      const response = await fetch(`http://127.0.0.1:8000/project/${id}`)
      const data = await response.json()
      setProjectId(data.id)
      setProjectData(data)
      setSelectedPartIndex(null)
      setActiveTool("none")
      setCurrentPoints([])
      setLassoPoints([])
      setIsDrawingLasso(false)
      setEditingPartId(null)
    } catch (error) {
      console.error('Error loading project:', error)
      alert('プロジェクトの読み込みに失敗しました。')
    } finally {
      setLoading(false)
    }
  }

  // Fetch projects on mount
  useEffect(() => {
    fetchProjects()
  }, [])

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setLoading(true)
    setProjectId(null)
    setProjectData(null)
    setSelectedPartIndex(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/project/create', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setProjectId(data.id)
      setProjectData(data)
      fetchProjects() // Refresh list
    } catch (error) {
      console.error('Error creating project:', error)
      alert('バックエンドに接続できません。Python サーバーが起動しているか確認してください。')
    } finally {
      setLoading(false)
    }
  }

  const handleAutoSegment = async () => {
    if (!projectId) return
    setLoading(true)
    setSelectedPartIndex(null)
    setActiveTool("none")
    try {
      const response = await fetch('http://127.0.0.1:8000/sam/auto-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: projectId }),
      })
      const masks = await response.json()
      setProjectData({ ...projectData, parts: masks })
      fetchProjects() // Refresh count
    } catch (error) {
      console.error('Auto segment error:', error)
      alert('自動候補生成に失敗しました。')
    } finally {
      setLoading(false)
    }
  }

  // Helper to extract client coordinates from either Mouse or Touch event
  const getClientCoords = (e: React.MouseEvent | React.TouchEvent) => {
    if ('touches' in e && e.touches.length > 0) {
      return { clientX: e.touches[0].clientX, clientY: e.touches[0].clientY };
    } else if ('changedTouches' in e && e.changedTouches.length > 0) {
      return { clientX: e.changedTouches[0].clientX, clientY: e.changedTouches[0].clientY };
    }
    return { clientX: (e as React.MouseEvent).clientX, clientY: (e as React.MouseEvent).clientY };
  };

  const getMappedCoords = (e: React.MouseEvent<HTMLElement> | React.TouchEvent<HTMLElement>) => {
    if (!projectData) return null;
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = projectData.width / rect.width;
    const scaleY = projectData.height / rect.height;
    const scale = Math.max(scaleX, scaleY);
    const displayedWidth = projectData.width / scale;
    const displayedHeight = projectData.height / scale;
    const offsetX = (rect.width - displayedWidth) / 2;
    const offsetY = (rect.height - displayedHeight) / 2;
    
    const { clientX, clientY } = getClientCoords(e);
    const clickX = clientX - rect.left - offsetX;
    const clickY = clientY - rect.top - offsetY;

    if (clickX < 0 || clickX > displayedWidth || clickY < 0 || clickY > displayedHeight) {
        return null;
    }

    const originalX = Math.round(clickX * scale);
    const originalY = Math.round(clickY * scale);

    return { 
        originalX, 
        originalY, 
        dispX: clickX + offsetX, 
        dispY: clickY + offsetY 
    };
  };

  const processInteractionStart = async (e: React.MouseEvent<HTMLImageElement> | React.TouchEvent<HTMLImageElement>, isShift: boolean = false) => {
    if ((activeTool !== "point" && activeTool !== "lasso") || !projectId || !projectData || loading) return;

    const coords = getMappedCoords(e);
    if (!coords) return;
    const { originalX, originalY, dispX, dispY } = coords;

    if (activeTool === "lasso") {
       // Start drawing lasso
       if (!isDrawingLasso) {
           setIsDrawingLasso(true);
           setLassoPoints([{x: originalX, y: originalY, dispX, dispY}]);
       } else {
           // Click again to finish drawing
           setIsDrawingLasso(false);
           // After drawing is done, wait for the user to click addition or subtraction button
       }
       return;
    }
    // Shift click for negative point
    const label = isShift ? 0 : 1;
    
    // Store point for display and calculation
    const newPoint = { x: originalX, y: originalY, label, dispX, dispY };
    const newPoints = [...currentPoints, newPoint];
    setCurrentPoints(newPoints);
    
    // Perform live preview prediction
    await updatePointPreview(newPoints);
  }

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => processInteractionStart(e, e.shiftKey);
  const handleTouchStart = (e: React.TouchEvent<HTMLImageElement>) => {
      // Prevent scrolling while drawing lasso or placing points on touch devices
      if (activeTool !== "none") {
         // Note: React's passive event listeners might warn here if we preventDefault,
         // but we need it to stop pull-to-refresh/scrolling. Can be added to a pure DOM ref later if needed.
      }
      processInteractionStart(e, false);
  };

  const processInteractionMove = (e: React.MouseEvent<HTMLImageElement> | React.TouchEvent<HTMLImageElement>) => {
      if (activeTool === "lasso" && isDrawingLasso) {
          const coords = getMappedCoords(e);
          if (!coords) return;
          const { originalX, originalY, dispX, dispY } = coords;
          
          // Add point if moved enough to avoid huge arrays
          if (lassoPoints.length === 0) return;
          const lastPt = lassoPoints[lassoPoints.length - 1];
          const dist = Math.hypot(dispX - lastPt.dispX, dispY - lastPt.dispY);
          
          if (dist > 5) { // 5px distance threshold
              setLassoPoints([...lassoPoints, {x: originalX, y: originalY, dispX, dispY}]);
          }
      }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLImageElement>) => processInteractionMove(e);
  const handleTouchMove = (e: React.TouchEvent<HTMLImageElement>) => {
      if (activeTool === "lasso" && isDrawingLasso) {
          // Prevent screen pan when drawing
          // Cannot reliably preventDefault here due to passive listeners in React.
      }
      processInteractionMove(e);
  };

  const handleContextMenu = (e: React.MouseEvent<HTMLImageElement>) => {
    if (activeTool === "point") {
      e.preventDefault(); // Prevent standard context menu
      // Treat right click as negative point
      const ev = { ...e, shiftKey: true } as any;
      handleImageClick(ev);
    } else if (activeTool === "lasso" && !isDrawingLasso && lassoPoints.length > 2) {
      e.preventDefault();
      // Right click after drawing lasso submits as negative
      submitLasso(0);
    }
  }

  // Previews the current mask without saving it to the project list permanently
  const updatePointPreview = async (pointsToUse: typeof currentPoints) => {
    if (pointsToUse.length === 0) return;
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/sam/predict-points', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          project_id: projectId,
          points: pointsToUse.map(p => [p.x, p.y]),
          labels: pointsToUse.map(p => p.label)
        }),
      })
      const masks = await response.json()
      setProjectData({ ...projectData, parts: masks })
      setSelectedPartIndex(masks.length - 1)
    } catch (error) {
      console.error('Point extract preview error:', error)
    } finally {
      setLoading(false)
    }
  }

  const submitLasso = async (label: number) => {
    if (lassoPoints.length < 3) {
       setLassoPoints([]);
       setIsDrawingLasso(false);
       return;
    }
    setLoading(true);
    try {
      // Determine if there's a target part to edit, or we're editing the preview
      let targetPartId = editingPartId;
      if (!targetPartId && selectedPartIndex !== null && projectData?.parts?.[selectedPartIndex]) {
          const part = projectData.parts[selectedPartIndex];
          if (part.id !== "preview") {
              targetPartId = part.id;
              setEditingPartId(part.id);
          }
      }

      const response = await fetch('http://127.0.0.1:8000/sam/predict-lasso', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          project_id: projectId,
          points: lassoPoints.map(p => [p.x, p.y]),
          label: label,
          target_part_id: targetPartId
        }),
      })
      const masks = await response.json()
      setProjectData({ ...projectData, parts: masks })
      setSelectedPartIndex(masks.length - 1)
    } catch (error) {
      console.error('Lasso extract error:', error)
      alert('投げ縄抽出に失敗しました。')
    } finally {
      setLassoPoints([]);
      setIsDrawingLasso(false);
      setLoading(false)
    }
  }

  const commitPoints = async () => {
    // Both point and lasso tools use preview.png
    // If we're editing an existing part, we should overwrite it.
    let targetPartId = editingPartId;
    if (!targetPartId && selectedPartIndex !== null && projectData?.parts?.[selectedPartIndex]) {
        const part = projectData.parts[selectedPartIndex];
        if (part.id !== "preview") {
            targetPartId = part.id;
        }
    }

    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/sam/commit-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            project_id: projectId,
            target_part_id: targetPartId
        }),
      })
      const masks = await response.json()
      setProjectData({ ...projectData, parts: masks })
      setSelectedPartIndex(masks.length - 1)
      fetchProjects() // Refresh list
    } catch (error) {
      console.error('Commit error:', error)
      alert('確定に失敗しました。')
    } finally {
      setCurrentPoints([]);
      setLassoPoints([]);
      setIsDrawingLasso(false);
      setEditingPartId(null);
      setActiveTool("none");
      setLoading(false);
    }
  }

  const handleRefine = async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/project/${projectId}/refine`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error("Refine failed");
      const masks = await response.json();
      setProjectData({ ...projectData, parts: masks });
      alert("線画補正が完了しました。");
      fetchProjects(); // Refresh list to update counts if any metadata changed
    } catch (error) {
       console.error("Refine error:", error);
       alert("補正に失敗しました。");
    } finally {
       setLoading(false);
    }
  }

  const handleExport = async () => {
    if (!projectId) return;
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/project/${projectId}/export`, {
        method: 'GET',
      });
      if (!response.ok) throw new Error("Export failed");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `paintbase_${projectId}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
       console.error("Export error:", error);
       alert("マスク画像の書き出しに失敗しました。");
    } finally {
       setLoading(false);
    }
  }

  const handleDeleteProject = async (id: string, name: string) => {
    if (!confirm(`プロジェクト「${name}」を削除してもよろしいですか？\nこの操作は取り消せません。`)) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/project/${id}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error("Delete failed");
      
      if (projectId === id) {
        setProjectId(null);
        setProjectData(null);
      }
      fetchProjects();
    } catch (error) {
       console.error("Delete error:", error);
       alert("削除に失敗しました。");
    } finally {
       setLoading(false);
    }
  }

  const handleFullExport = async (id: string, name: string) => {
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8000/project/${id}/export-full`, {
        method: 'GET',
      });
      if (!response.ok) throw new Error("Full export failed");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `paintbase_full_${name}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
       console.error("Full export error:", error);
       alert("プロジェクトのエクスポートに失敗しました。");
    } finally {
       setLoading(false);
    }
  }

  const handleImportProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/project/import', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error("Import failed");
      const data = await response.json();
      setProjectId(data.id);
      setProjectData(data);
      fetchProjects();
      alert("プロジェクトをインポートしました。");
    } catch (error) {
      console.error('Import error:', error);
      alert('インポートに失敗しました。正しいプロジェクトZIPか確認してください。');
    } finally {
      setLoading(false);
      e.target.value = ""; // Clear input
    }
  }

  const clearPoints = () => {
    setCurrentPoints([]);
    
    // Remove preview part from UI
    if (projectData?.parts && projectData.parts.length > 0) {
        const lastPart = projectData.parts[projectData.parts.length - 1];
        if (lastPart.id === "preview") {
             const newParts = projectData.parts.slice(0, -1);
             setProjectData({ ...projectData, parts: newParts });
             setSelectedPartIndex(null);
        }
    }
    setActiveTool("none");
    setLassoPoints([]);
    setIsDrawingLasso(false);
    setEditingPartId(null);
  }

  return (
    <div className="app-container">
      {/* Upper Toolbar */}
      <header className="toolbar">
        <div style={{ fontWeight: 'bold', marginRight: '24px' }}>PaintBase</div>
        <button 
          className="button button-secondary" 
          disabled={!projectId || loading}
          onClick={handleAutoSegment}
        >
          {loading && activeTool === "none" && projectId ? '計算中...' : '自動候補生成'}
        </button>
        <button 
          className={`button button-secondary ${activeTool === 'point' ? 'active' : ''}`} 
          disabled={!projectId}
          onClick={() => {
            setActiveTool(activeTool === 'point' ? 'none' : 'point');
            setCurrentPoints([]); // Clear points when toggling tool
            setEditingPartId(null);
          }}
          style={{ background: activeTool === 'point' ? 'var(--accent-color)' : '', color: activeTool === 'point' ? '#fff' : '' }}
        >
          点追加
        </button>
        {activeTool === 'point' && currentPoints.length > 0 && (
          <div style={{ display: 'flex', gap: '4px' }}>
             <span style={{ fontSize: '0.8rem', alignSelf: 'center', marginLeft: '8px' }}>
                左クリック:追加 / 右・Shift:除外
             </span>
          </div>
        )}
        <button 
          className={`button button-secondary ${activeTool === 'lasso' ? 'active' : ''}`} 
          disabled={!projectId}
          onClick={() => {
            setActiveTool(activeTool === 'lasso' ? 'none' : 'lasso');
            setLassoPoints([]);
            setIsDrawingLasso(false);
            setEditingPartId(null);
          }}
          style={{ background: activeTool === 'lasso' ? 'var(--accent-color)' : '', color: activeTool === 'lasso' ? '#fff' : '' }}
        >
          投げ縄修正
        </button>
        {activeTool === 'lasso' && lassoPoints.length > 0 && !isDrawingLasso && (
          <div style={{ display: 'flex', gap: '4px' }}>
             <button className="button" style={{ background: '#44bb44' }} onClick={() => submitLasso(1)}>追加</button>
             <button className="button" style={{ background: '#ff4444' }} onClick={() => submitLasso(0)}>除外 (右クリック)</button>
             <button className="button button-secondary" onClick={() => {
                 setLassoPoints([]);
                 setIsDrawingLasso(false);
             }}>クリア</button>
             {(editingPartId || (selectedPartIndex !== null && projectData?.parts?.[selectedPartIndex]?.id !== "preview")) && (
                <span style={{ fontSize: '0.8rem', alignSelf: 'center', marginLeft: '8px', color: '#ffbb44' }}>
                   *既存パーツを編集します
                </span>
             )}
          </div>
        )}
        
        {/* Universal Commit UI for both point and lasso tools when preview exists */}
        {projectData?.parts?.[projectData.parts.length - 1]?.id === "preview" && (
           <div style={{ display: 'flex', gap: '4px', marginLeft: '16px' }}>
             <button className="button" style={{ background: '#44bb44' }} onClick={commitPoints}>修正を確定</button>
             <button className="button button-secondary" onClick={clearPoints}>修正を破棄</button>
           </div>
        )}
        <button 
          className="button button-secondary" 
          disabled={!projectId || loading}
          onClick={handleRefine}
          style={{ marginLeft: '16px' }}
        >
          {loading && activeTool === "none" && projectId ? '補正中...' : '線画補正'}
        </button>

        <div style={{ display: 'flex', alignItems: 'center', marginRight: '16px', gap: '8px', fontSize: '0.8rem' }}>
          <span>BG明るさ:</span>
          <input 
            type="range" 
            min="0" 
            max="100" 
            value={bgBrightness} 
            onChange={(e) => setBgBrightness(parseInt(e.target.value))}
            style={{ width: '80px' }}
          />
        </div>
        <button className="button" disabled={!projectId || loading} onClick={handleExport} title="現在のプロジェクトのマスクをZIPで書き出し">
          {loading && activeTool === "none" && projectId ? '処理中...' : 'マスクの書き出し'}
        </button>
      </header>

      <main className="main-content">
        {/* Left Pane: Project Info */}
        <aside className="sidebar">
          <div className="panel-header">プロジェクト</div>
          
          <div className="project-list-container">
            {projectId && projectData && (
              <div style={{ marginBottom: '16px', paddingBottom: '16px', borderBottom: '1px solid var(--border-color)' }}>
                <p style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
                  <strong>{projectData.name}</strong>
                </p>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                  <p>ID: {projectId.slice(0, 8)}...</p>
                  <p>Size: {projectData.width} x {projectData.height}</p>
                </div>
                <button 
                  className="button button-secondary" 
                  style={{ width: '100%', marginTop: '12px', fontSize: '0.8rem' }}
                  onClick={() => setProjectId(null)}
                >
                  プロジェクトを閉じる
                </button>
              </div>
            )}

            {!projectId && (
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginBottom: '24px' }}>
                <p style={{ marginBottom: '16px' }}>画像を選択して開始</p>
                <input
                  type="file"
                  id="fileInput"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                  accept="image/*"
                />
                <button
                  className="button"
                  onClick={() => document.getElementById('fileInput')?.click()}
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? '読み込み中...' : '新規プロジェクト'}
                </button>
              </div>
            )}

            <div style={{ fontSize: '0.8rem', fontWeight: 'bold', marginBottom: '8px', color: 'var(--text-secondary)' }}>
              保存されたプロジェクト
            </div>
            
            <div className="project-list">
              {projectsList.length === 0 ? (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>履歴なし</p>
              ) : (
                projectsList.map((p) => (
                  <div 
                    key={p.id} 
                    className={`project-item ${projectId === p.id ? 'active' : ''}`}
                    onClick={() => projectId !== p.id && loadProject(p.id)}
                    style={{ position: 'relative' }}
                  >
                    <div className="project-item-name">{p.name}</div>
                    <div className="project-item-meta">{p.parts_count} parts | {p.width}x{p.height}</div>
                    
                    <div className="project-item-actions">
                      <button 
                        className="action-button" 
                        onClick={(e) => { e.stopPropagation(); handleFullExport(p.id, p.name); }}
                        title="全体をZIP保存"
                      >
                        💾
                      </button>
                      <button 
                        className="action-button action-delete" 
                        onClick={(e) => { e.stopPropagation(); handleDeleteProject(p.id, p.name); }}
                        title="削除"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            <div style={{ marginTop: 'auto', paddingTop: '16px' }}>
              <input
                type="file"
                id="importInput"
                style={{ display: 'none' }}
                onChange={handleImportProject}
                accept=".zip"
              />
              <button
                className="button button-secondary"
                onClick={() => document.getElementById('importInput')?.click()}
                disabled={loading}
                style={{ width: '100%', fontSize: '0.8rem' }}
              >
                プロジェクトを読み込む (.zip)
              </button>
            </div>
          </div>
        </aside>

        {/* Central Canvas */}
        <section 
          className="canvas-area" 
          style={{ backgroundColor: `rgb(${bgBrightness * 2.55}, ${bgBrightness * 2.55}, ${bgBrightness * 2.55})` }}
        >
          {projectId && projectData ? (
            <div style={{ position: 'relative', boxShadow: '0 0 20px rgba(0,0,0,0.3)', lineHeight: 0 }}>
              <img
                src={`http://127.0.0.1:8000/projects/${projectId}/${projectData.source_image}`}
                alt="Main"
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '100%', 
                  objectFit: 'contain', 
                  display: 'block',
                  cursor: activeTool === 'point' ? (loading ? 'wait' : 'crosshair') : (activeTool === 'lasso' ? 'crosshair' : 'default'),
                  userSelect: 'none'
                }}
                onClick={handleImageClick}
                onMouseMove={handleMouseMove}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onContextMenu={handleContextMenu}
              />
              {selectedPartIndex !== null && projectData.parts?.[selectedPartIndex] && (
                <img
                  src={`http://127.0.0.1:8000/projects/${projectId}/${projectData.parts[selectedPartIndex].mask_path}?t=${Date.now()}`}
                  alt="Mask Overlay"
                  style={{ 
                    position: 'absolute', 
                    top: 0, 
                    left: 0, 
                    width: '100%', 
                    height: '100%', 
                    objectFit: 'contain',
                    pointerEvents: 'none',
                    mixBlendMode: 'plus-lighter',
                    opacity: 0.6,
                    filter: 'invert(40%) sepia(100%) saturate(400%) hue-rotate(180deg)', // Cyan overlay
                  }}
                />
              )}
              {/* Lasso path SVG */}
              {activeTool === 'lasso' && lassoPoints.length > 0 && (
                <svg
                  style={{
                    position: 'absolute',
                    top: 0, left: 0,
                    width: '100%', height: '100%',
                    pointerEvents: 'none',
                    zIndex: 20
                  }}
                >
                  <polyline 
                    points={lassoPoints.map(p => `${p.dispX},${p.dispY}`).join(' ')}
                    fill={isDrawingLasso ? "none" : "rgba(0, 255, 255, 0.2)"}
                    stroke="#00ffff"
                    strokeWidth="2"
                    strokeDasharray={isDrawingLasso ? "4 4" : "none"}
                  />
                  {!isDrawingLasso && lassoPoints.length > 2 && (
                    <polygon 
                        points={lassoPoints.map(p => `${p.dispX},${p.dispY}`).join(' ')}
                        fill="rgba(0, 255, 255, 0.3)"
                    />
                  )}
                </svg>
              )}
              {/* Point markers */}
              {activeTool === 'point' && currentPoints.map((pt, i) => (
                <div 
                  key={i} 
                  style={{
                    position: 'absolute',
                    left: `${pt.dispX}px`,
                    top: `${pt.dispY}px`,
                    width: '10px',
                    height: '10px',
                    backgroundColor: pt.label === 1 ? '#00ffff' : '#ff0000',
                    borderRadius: '50%',
                    transform: 'translate(-50%, -50%)',
                    pointerEvents: 'none',
                    boxShadow: '0 0 4px rgba(0,0,0,0.8)',
                    zIndex: 10
                  }}
                />
              ))}
            </div>
          ) : (
            <div style={{ color: 'var(--text-secondary)' }}>キャンバス (画像未読み込み)</div>
          )}
        </section>

        {/* Right Pane: Parts List */}
        <aside className="sidebar">
          <div className="panel-header">パーツ</div>
          <div style={{ flex: 1, padding: '16px', overflowY: 'auto' }}>
            {projectData?.parts && projectData.parts.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {projectData.parts.map((part: any, idx: number) => (
                  <div 
                    key={part.id} 
                    onClick={() => setSelectedPartIndex(idx)}
                    style={{ 
                      padding: '8px', 
                      background: selectedPartIndex === idx ? 'var(--accent-color)' : 'var(--bg-secondary)', 
                      color: selectedPartIndex === idx ? '#fff' : 'inherit',
                      borderRadius: '4px',
                      fontSize: '0.8rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer'
                    }}
                  >
                    <span>パーツ {idx + 1}</span>
                    <span style={{ color: selectedPartIndex === idx ? '#fff' : 'var(--text-secondary)' }}>
                      IoU: {part.predicted_iou.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--text-secondary)' }}>
                {projectId ? 'パーツがありません' : '画像が読み込まれていません'}
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
