import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as echarts from "echarts/core";
import { LineChart, BarChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
} from "echarts/components";
import { SVGRenderer } from "echarts/renderers";
import "./style.css";

echarts.use([
  LineChart,
  BarChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
  SVGRenderer,
]);
type Data = Record<string, any>;
type Meta = {
  jobs: Data[];
  environments: Data[];
  projects: Data[];
  devices: Data[];
  bootstrap: Data;
};
let csrf = "";
async function api(path: string, body?: any, method?: string): Promise<any> {
  const headers: Record<string, string> = {};
  if (body !== undefined && !(body instanceof FormData))
    headers["Content-Type"] = "application/json";
  if (body !== undefined) headers["X-KTL-CSRF"] = csrf;
  const response = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers,
    credentials: "same-origin",
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/session")
      window.dispatchEvent(new Event("ktlab-session-expired"));
    throw new Error(
      result.error || result.detail || `请求失败 (${response.status})`,
    );
  }
  return result;
}
const pretty = (x: any) => JSON.stringify(x, null, 2);
const ref = (x: any) => (typeof x === "string" ? x : x?.asset || "");
const short = (x: any) =>
  String(x || "—")
    .replace(/@[^@]*$/, "")
    .replace(/^production\//, "");
const num = (x: any, unit = "") =>
  x == null || !Number.isFinite(Number(x))
    ? "—"
    : `${Number(x).toLocaleString("en", { maximumSignificantDigits: 5 })}${unit}`;
const bytes = (x: number) =>
  x >= 1024 ** 2
    ? `${(x / 1024 ** 2).toFixed(1)} MB`
    : x >= 1024
      ? `${(x / 1024).toFixed(1)} KB`
      : `${x} B`;
const when = (x: number) =>
  x
    ? new Date(x * 1000).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
const names: Data = {
  draft: "草稿",
  prechecking: "预检中",
  ready: "可提交",
  invalid: "预检失败",
  queued: "排队",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
  cancelling: "取消中",
  cancelled: "已取消",
  passed: "通过",
  success: "成功",
  correctness_error: "数值失败",
  compile_error: "编译失败",
  runtime_error: "运行失败",
  not_executed: "未执行",
  available: "可用",
  probing: "探测中",
  unavailable: "不可用",
};
function Status({ value }: { value: string }) {
  return (
    <span className={`status status-${value}`}>
      {names[value] || value || "—"}
    </span>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}
function Json({ value }: { value: any }) {
  return <pre className="json">{pretty(value)}</pre>;
}
function Field({
  label,
  children,
  wide = false,
}: {
  label: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <label className={wide ? "field wide" : "field"}>
      <span>{label}</span>
      {React.Children.map(children, (child) =>
        React.isValidElement(child)
          ? React.cloneElement(child as React.ReactElement<any>, {
              "aria-label": label,
            })
          : child,
      )}
    </label>
  );
}
function ParameterTable({ value }: { value: any }) {
  if (!value || typeof value !== "object") return <span>—</span>;
  return (
    <dl className="parameters">
      {Object.entries(value).map(([key, val]) => (
        <React.Fragment key={key}>
          <dt>{key}</dt>
          <dd>
            {typeof val === "object" ? (
              <code>{JSON.stringify(val)}</code>
            ) : (
              String(val)
            )}
          </dd>
        </React.Fragment>
      ))}
    </dl>
  );
}
function Chart({
  option,
  label,
  height = 350,
}: {
  option: any;
  label: string;
  height?: number;
}) {
  const div = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);
  useEffect(() => {
    if (!div.current) return;
    const chart = echarts.init(div.current, undefined, { renderer: "svg" });
    instance.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(div.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, []);
  useEffect(() => {
    instance.current?.setOption(option, {
      notMerge: false,
      replaceMerge: ["series"],
    });
  }, [option]);
  function download() {
    const url = instance.current?.getDataURL({ type: "svg" });
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${label}.svg`;
    a.click();
  }
  return (
    <div className="chart-wrap">
      <button
        className="chart-download text-button"
        onClick={download}
        title="下载当前图表 SVG"
      >
        下载图表 ↗
      </button>
      <div
        className="chart"
        ref={div}
        style={{ height }}
        role="img"
        aria-label={label}
      />
    </div>
  );
}
const baseChart = {
  animation: false,
  color: ["#2563eb", "#11958b", "#9b59b6", "#d97706", "#d65a68", "#537385"],
  textStyle: { fontFamily: "system-ui, sans-serif", fontSize: 13 },
  tooltip: { trigger: "axis", renderMode: "richText", confine: true },
  legend: { top: 14, left: 18, right: 110, type: "scroll", itemWidth: 18 },
  grid: { left: 70, right: 28, top: 62, bottom: 75 },
  dataZoom: [
    { type: "inside" },
    {
      type: "slider",
      height: 18,
      bottom: 15,
      borderColor: "#dbe3ed",
      showDetail: false,
    },
  ],
};

function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [online, setOnline] = useState(true);
  const [route, setRoute] = useState(location.hash.slice(1) || "workspace");
  const [meta, setMeta] = useState<Meta>({
    jobs: [],
    environments: [],
    projects: [],
    devices: [],
    bootstrap: {},
  });
  const navigate = (where: string) => {
    location.hash = where;
  };
  const fail = (e: any) => {
    if (e instanceof TypeError) {
      setOnline(false);
      return;
    }
    setError(e.message || String(e));
  };
  useEffect(() => {
    const fn = () => {
      setRoute(location.hash.slice(1) || "workspace");
      setError("");
    };
    window.addEventListener("hashchange", fn);
    return () => window.removeEventListener("hashchange", fn);
  }, []);
  useEffect(() => {
    api("/api/session")
      .then((x) => {
        csrf = x.csrf || "";
        setAuthenticated(x.authenticated);
      })
      .catch(fail);
  }, []);
  useEffect(() => {
    const expired = () => {
      csrf = "";
      setAuthenticated(false);
    };
    window.addEventListener("ktlab-session-expired", expired);
    return () => window.removeEventListener("ktlab-session-expired", expired);
  }, []);
  async function refresh() {
    const [bootstrap, jobs, environments, projects] = await Promise.all([
      api("/api/bootstrap"),
      api("/api/jobs"),
      api("/api/environments"),
      api("/api/projects"),
    ]);
    setMeta((old) => ({
      ...old,
      bootstrap,
      jobs: jobs.jobs,
      environments: environments.environments,
      projects: projects.projects,
    }));
  }
  useEffect(() => {
    if (!authenticated) return;
    refresh().catch(fail);
    api("/api/devices")
      .then((x) => setMeta((m) => ({ ...m, devices: x.devices })))
      .catch(fail);
    const timer = setInterval(() => {
      Promise.all([api("/api/jobs"), api("/api/bootstrap")])
        .then(([j, b]) => {
          setMeta((m) => ({ ...m, jobs: j.jobs, bootstrap: b }));
          setOnline(true);
        })
        .catch((e) => {
          if (e instanceof TypeError) setOnline(false);
          else fail(e);
        });
    }, 2500);
    return () => clearInterval(timer);
  }, [authenticated]);
  useEffect(() => {
    if (
      !authenticated ||
      !meta.environments.some((e) => e.probe?.status === "probing")
    )
      return;
    const timer = setTimeout(
      () =>
        api("/api/environments")
          .then((x) => setMeta((m) => ({ ...m, environments: x.environments })))
          .catch(fail),
      1500,
    );
    return () => clearTimeout(timer);
  }, [authenticated, meta.environments]);
  async function login(e: React.FormEvent) {
    e.preventDefault();
    try {
      const x = await api("/api/session", { token });
      csrf = x.csrf;
      setToken("");
      setAuthenticated(true);
      setError("");
    } catch (e) {
      fail(e);
    }
  }
  if (authenticated === null)
    return (
      <div className="login">
        <h1>KernelTuningLab</h1>
        <p>连接实验工作台…</p>
      </div>
    );
  if (!authenticated)
    return (
      <div className="login">
        <div className="brand-mark">K</div>
        <h1>连接实验工作台</h1>
        <p>使用远端工作区的访问令牌登录。</p>
        <form onSubmit={login}>
          <Field label="访问令牌">
            <input
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              autoComplete="off"
              autoFocus
              required
            />
          </Field>
          <button className="primary" type="submit">
            连接工作区 →
          </button>
        </form>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <small>令牌位置：工作区 / access-token</small>
      </div>
    );
  const page = route.split("/")[0];
  const title: Data = {
    workspace: "实验工作区",
    new: "新建实验",
    queue: "任务队列",
    assets: "算子与算法",
    analysis: "结果分析",
    devices: "设备与环境",
    job: "实验详情",
    history: "历史实验",
  };
  const nav = [
    ["workspace", "▦", "实验工作区"],
    ["new", "＋", "新建实验"],
    ["queue", "≡", "任务队列"],
    ["assets", "◇", "算子与算法"],
    ["analysis", "↗", "结果分析"],
    ["devices", "◉", "设备与环境"],
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#workspace">
          <b className="brand-mark">K</b>
          <span>
            KernelTuningLab<small>实验工作台</small>
          </span>
        </a>
        <nav>
          {nav.map(([key, icon, text]) => (
            <a
              href={"#" + key}
              key={key}
              className={page === key ? "active" : ""}
            >
              <span className="nav-icon" aria-hidden>
                {icon}
              </span>
              {text}
            </a>
          ))}
        </nav>
        <div className="server-status">
          <i className={meta.bootstrap.worker?.alive ? "dot live" : "dot"} />
          <div>
            远端工作区
            <small>
              {meta.bootstrap.workspace_name || "连接中"} · v
              {meta.bootstrap.version || "—"}
            </small>
          </div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <span>
            工作台 <span className="breadcrumb">/ {title[page] || "实验"}</span>
          </span>
          <span className="connection">
            <i className={online ? "dot live" : "dot"} />{" "}
            {online ? "工作区在线" : "正在重连"}
          </span>
        </header>
        <div className="page">
          {error && (
            <div className="error-banner" role="alert">
              {error}
              <button className="text-button" onClick={() => setError("")}>
                关闭
              </button>
            </div>
          )}
          {page === "new" ? (
            <NewExperiment
              meta={meta}
              draftId={route.split("/")[1]}
              refresh={refresh}
              navigate={navigate}
              fail={fail}
            />
          ) : page === "job" ? (
            <JobDetail
              id={route.split("/")[1]}
              meta={meta}
              refresh={refresh}
              navigate={navigate}
              fail={fail}
            />
          ) : page === "assets" ? (
            <Assets meta={meta} />
          ) : page === "devices" ? (
            <Devices
              meta={meta}
              refresh={refresh}
              setMeta={setMeta}
              fail={fail}
            />
          ) : page === "analysis" ? (
            <Analysis meta={meta} fail={fail} />
          ) : page === "history" ? (
            <History id={route.split("/")[1]} fail={fail} />
          ) : (
            <Workspace
              meta={meta}
              queue={page === "queue"}
              navigate={navigate}
              fail={fail}
            />
          )}
        </div>
      </main>
    </div>
  );
}

function Workspace({
  meta,
  queue,
  navigate,
  fail,
}: {
  meta: Meta;
  queue: boolean;
  navigate: (x: string) => void;
  fail: (e: any) => void;
}) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [environment, setEnvironment] = useState("");
  const jobs = meta.jobs.filter(
    (j) =>
      (!queue ||
        ["queued", "running", "prechecking", "cancelling"].includes(j.state)) &&
      (!status || j.state === status) &&
      (!environment || j.environment === environment) &&
      `${j.name} ${j.spec?.operator} ${j.spec?.algorithms?.map(ref).join(" ")}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const running = meta.jobs.filter((j) => j.state === "running").length;
  const queued = meta.jobs.filter((j) => j.state === "queued").length;
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">REMOTE EXPERIMENTS</div>
          <h1>{queue ? "任务队列" : "实验工作区"}</h1>
          <p>
            {running} 个运行中 · {queued} 个排队 · 源码与完整记录保存在服务器
          </p>
        </div>
        <button className="primary" onClick={() => navigate("new")}>
          ＋ 新建实验
        </button>
      </div>
      <div className="toolbar">
        <input
          aria-label="搜索实验"
          placeholder="搜索名称、算子或算法"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          aria-label="实验状态"
          value={status}
          onChange={(e) => setStatus(e.target.value)}
        >
          <option value="">全部状态</option>
          {Object.entries(names)
            .filter(([k]) =>
              [
                "draft",
                "prechecking",
                "ready",
                "invalid",
                "queued",
                "running",
                "completed",
                "failed",
                "interrupted",
                "cancelled",
              ].includes(k),
            )
            .map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
        </select>
        <select
          aria-label="执行环境筛选"
          value={environment}
          onChange={(e) => setEnvironment(e.target.value)}
        >
          <option value="">全部环境</option>
          {meta.environments.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label}
            </option>
          ))}
        </select>
        <a className="text-button" href="#history">
          查看历史结果 ↗
        </a>
      </div>
      <section className="panel">
        <div className="table-scroll">
          <table className="experiments">
            <thead>
              <tr>
                <th>实验</th>
                <th>状态</th>
                <th>执行环境</th>
                <th>算子 / 方法</th>
                <th>预算</th>
                <th>更新时间</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td>
                    <button
                      className="row-link"
                      onClick={() => navigate("job/" + j.id)}
                    >
                      {j.name}
                    </button>
                    <small>
                      {j.spec?.dtype} · seed {j.spec?.seed ?? "—"}
                    </small>
                  </td>
                  <td>
                    <Status value={j.state} />
                    {j.resource_blocked && (
                      <small className="amber">资源忙 · 等待重试</small>
                    )}
                  </td>
                  <td>
                    {meta.environments.find((e) => e.id === j.environment)
                      ?.label || j.environment}
                    <small>{j.gpu ? j.gpu.slice(0, 16) + "…" : "CPU"}</small>
                  </td>
                  <td>
                    {short(j.spec?.operator)}
                    <small>
                      {(j.spec?.algorithms || [])
                        .map((a: any) => short(ref(a)))
                        .join(" / ")}
                    </small>
                  </td>
                  <td>
                    {j.planned
                      ? num(j.planned.max_observations)
                      : num(j.spec?.budget?.max_evaluations)}{" "}
                    次
                    <small>
                      {num(j.spec?.budget?.max_wall_time_s, " s / 方法")}
                    </small>
                  </td>
                  <td className="nowrap">{when(j.updated)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!jobs.length && (
          <Empty>
            {queue
              ? "当前没有运行或排队的任务。"
              : "还没有实验。从模板或上传项目开始。"}
            <button className="primary" onClick={() => navigate("new")}>
              新建实验 →
            </button>
          </Empty>
        )}
      </section>
      <p className="subtle">
        {jobs.length} 个实验
        {queue ? " · Worker 按顺序执行，资源忙的任务等待手动重试。" : ""}
      </p>
    </>
  );
}

function NewExperiment({
  meta,
  draftId,
  refresh,
  navigate,
  fail,
}: {
  meta: Meta;
  draftId?: string;
  refresh: () => Promise<void>;
  navigate: (x: string) => void;
  fail: (e: any) => void;
}) {
  const initial = meta.bootstrap.templates?.[0]?.spec || {};
  const [spec, setSpec] = useState<Data>(initial);
  const [text, setText] = useState(pretty(initial));
  const [advanced, setAdvanced] = useState(false);
  const [environment, setEnvironment] = useState("core");
  const [gpu, setGpu] = useState("");
  const [action, setAction] = useState("run");
  const [projectKey, setProjectKey] = useState("");
  const [source, setSource] = useState("");
  const [ack, setAck] = useState(false);
  const [draft, setDraft] = useState<Data | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [preview, setPreview] = useState<Data | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const selectedProject = meta.projects.find(
    (p) => `${p.id}/${p.version}` === projectKey,
  );
  const env = meta.environments.find((e) => e.id === environment);
  const assets: Data[] = env?.probe?.assets || [];
  const uploadedAssets = (selectedProject?.files || []).flatMap((f: Data) =>
    (f.assets || []).map((a: Data) => ({
      ...a,
      ref: `${a.id}@${a.version}`,
      kind:
        String(a["constructor"]) === "KernelDefinition"
          ? "kernel"
          : String(a["constructor"]) === "OperatorDefinition"
            ? "operator"
            : "algorithm",
      file: f.path,
    })),
  );
  const catalogue = [
    ...assets,
    ...uploadedAssets.filter((a: Data) => !assets.some((x) => x.ref === a.ref)),
  ];
  const operators = catalogue.filter((a) => a.kind === "operator");
  const operator = catalogue.find(
    (a) => a.kind === "operator" && a.ref === spec.operator,
  );
  const kernels = catalogue.filter(
    (a) =>
      a.kind === "kernel" &&
      (!a.operator || a.operator === spec.operator) &&
      (!a.runtime || (a.runtime === "python") === (env?.device === "cpu")),
  );
  const kernel = catalogue.find((a) => a.ref === ref(spec.kernels?.[0]));
  const algorithms = catalogue.filter((a) => a.kind === "algorithm");
  function update(value: Data) {
    setSpec(value);
    setText(pretty(value));
    setNotice("");
  }
  useEffect(() => {
    if (draftId)
      api("/api/jobs/" + draftId)
        .then((j) => {
          setDraft(j);
          update(j.spec);
          setEnvironment(j.environment);
          setGpu(j.gpu || "");
          setAction(j.action);
          setAdvanced(j.action === "research");
          setProjectKey(
            j.project ? `${j.project.id}/${j.project.version}` : "",
          );
        })
        .catch(fail);
  }, [draftId]);
  useEffect(() => {
    if (!Object.keys(spec).length && meta.bootstrap.templates?.length)
      update(initial);
  }, [meta.bootstrap.templates]);
  async function upload(incoming: FileList | File[] | null) {
    if (!incoming?.length) return;
    setUploadBusy(true);
    try {
      const form = new FormData();
      for (const file of Array.from(incoming)) form.append("files", file);
      const p = await api("/api/projects", form);
      await refresh();
      setProjectKey(`${p.id}/${p.version}`);
      setAck(false);
      setSource("");
      const found = p.files.find((f: Data) => f.experiment);
      if (found) {
        update(found.experiment);
        setAction(found.experiment.kind === "research" ? "research" : "run");
        setAdvanced(found.experiment.kind === "research");
      } else {
        const entry = p.files.find(
          (f: Data) => f.path.endsWith(".py") && f.registration_hook,
        );
        if (entry)
          update({
            ...spec,
            plugins: [...new Set([...(spec.plugins || []), entry.path])],
          });
      }
      setNotice("项目已上传 · 静态检查完成");
    } catch (e) {
      fail(e);
    } finally {
      setUploadBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }
  async function confirm() {
    if (!selectedProject) return;
    try {
      await api(
        `/api/projects/${selectedProject.id}/${selectedProject.version}/trust`,
        { acknowledged: ack, source },
      );
      await refresh();
      setNotice("此代码版本已确认可信");
    } catch (e) {
      fail(e);
    }
  }
  async function save(precheck = false) {
    setBusy(true);
    try {
      const raw = advanced ? JSON.parse(text) : spec;
      const body = {
        spec: raw,
        environment,
        gpu: env?.device === "cuda" ? gpu : null,
        action,
        project: selectedProject
          ? { id: selectedProject.id, version: selectedProject.version }
          : null,
      };
      let saved = draft
        ? await api(
            "/api/jobs/" + draft.id,
            { ...body, revision: draft.revision },
            "PUT",
          )
        : await api("/api/jobs", body);
      setDraft(saved);
      await refresh();
      if (precheck) {
        await api(`/api/jobs/${saved.id}/precheck`, {});
        navigate("job/" + saved.id);
      } else setNotice("草稿已保存");
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  function template(t: Data) {
    update(structuredClone(t.spec));
    setDraft(null);
    setProjectKey("");
    setAdvanced(false);
    setAction("run");
    const matching = meta.environments.find((e) => e.profile === t.profile);
    if (matching) setEnvironment(matching.id);
  }
  function changeOperator(value: string) {
    const op = catalogue.find((a) => a.ref === value);
    const k = catalogue.find(
      (a) =>
        a.kind === "kernel" &&
        a.operator === value &&
        (a.runtime === "python") === (env?.device === "cpu"),
    );
    update({
      ...spec,
      operator: value,
      kernels: k ? [{ asset: k.ref }] : spec.kernels,
      dtype: op?.dtypes?.[0] || spec.dtype,
      parameters: op?.parameters || {},
      shapes: [
        op?.validation_shapes?.[0] ||
          Object.fromEntries(
            (op?.shape_dims || ["M", "N", "K"]).map((x: string) => [x, 16]),
          ),
      ],
    });
  }
  const dims = spec.shapes?.[0]?.dims || spec.shapes?.[0] || {};
  const planned =
    action === "research"
      ? null
      : (spec.kernels?.length || 0) *
        (spec.algorithms?.length || 0) *
        (spec.shapes?.length || 0);
  const needsTrust =
    selectedProject?.files.some((f: Data) => f.path.endsWith(".py")) &&
    !selectedProject.trusted;
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">CREATE EXPERIMENT</div>
          <h1>{draft ? "编辑实验" : "新建实验"}</h1>
          <p>选择服务器资产，或上传自己的算子与搜索器。</p>
        </div>
        <button onClick={() => save()} disabled={busy}>
          保存草稿
        </button>
      </div>
      <div className="template-row">
        {meta.bootstrap.templates?.map((t: Data) => (
          <button key={t.id} onClick={() => template(t)}>
            {t.name} 模板
          </button>
        ))}
        <select
          aria-label="复用历史配置"
          value=""
          onChange={(e) => navigate("new/" + e.target.value)}
        >
          <option value="">复用草稿配置…</option>
          {meta.jobs
            .filter((j) => ["draft", "ready", "invalid"].includes(j.state))
            .map((j) => (
              <option key={j.id} value={j.id}>
                {j.name}
              </option>
            ))}
        </select>
      </div>
      <div className="create-layout">
        <div className="create-main">
          <section className="panel form-panel">
            <div className="section-heading">
              <h2>项目与文件</h2>
              <span className="subtle">上传后不会自动导入 Python</span>
            </div>
            <div
              className={"dropzone" + (uploadBusy ? " loading" : "")}
              role="button"
              tabIndex={0}
              aria-label="上传项目"
              onClick={() => fileInput.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter") fileInput.current?.click();
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                upload(e.dataTransfer.files);
              }}
            >
              <b className="upload-icon">↑</b>
              <strong>
                {uploadBusy ? "正在上传…" : "拖拽文件到这里，或点击选择"}
              </strong>
              <span>
                Python 插件、实验 JSON 或 ZIP 项目 · 最大{" "}
                {bytes(meta.bootstrap.limits?.upload_bytes || 32 * 1024 ** 2)}
              </span>
              <input
                ref={fileInput}
                type="file"
                multiple
                accept=".py,.json,.zip,.toml,.txt"
                onChange={(e) => upload(e.target.files)}
                hidden
              />
            </div>
            {!!meta.projects.length && (
              <Field label="已上传项目版本">
                <select
                  value={projectKey}
                  onChange={(e) => {
                    setProjectKey(e.target.value);
                    setAck(false);
                  }}
                >
                  <option value="">使用服务器资产</option>
                  {meta.projects.map((p) => (
                    <option
                      key={`${p.id}/${p.version}`}
                      value={`${p.id}/${p.version}`}
                    >
                      {p.name} · {p.version.slice(0, 10)}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            {selectedProject && (
              <>
                <div className="file-list">
                  {selectedProject.files.map((f: Data) => (
                    <button
                      key={f.path}
                      className="file-row"
                      onClick={async () => {
                        try {
                          setPreview({
                            ...f,
                            ...(await api(
                              `/api/projects/${selectedProject.id}/${selectedProject.version}/file?path=${encodeURIComponent(f.path)}`,
                            )),
                          });
                        } catch (e) {
                          fail(e);
                        }
                      }}
                    >
                      <span>
                        <b>{f.path}</b>
                        <small>
                          {f.assets?.length
                            ? `${f.assets.length} 个静态资产声明`
                            : "源文件"}
                          {f.errors?.length ? " · 解析失败" : ""}
                        </small>
                      </span>
                      <span>
                        {bytes(f.bytes)}
                        <small className="mono" title={f.sha256}>
                          SHA {f.sha256.slice(0, 12)}
                        </small>
                      </span>
                    </button>
                  ))}
                </div>
                {selectedProject.files.flatMap((f: Data) =>
                  f.errors.map((e: string) => (
                    <p key={f.path + e} className="error">
                      {f.path}：{e}
                    </p>
                  )),
                )}
                <details className="disclosure">
                  <summary>依赖与发现的资产</summary>
                  <Json
                    value={selectedProject.files
                      .filter(
                        (f: Data) => f.assets?.length || f.dependency_text,
                      )
                      .map((f: Data) => ({
                        file: f.path,
                        assets: f.assets,
                        imports: f.imports,
                        requirements: f.dependency_text,
                      }))}
                  />
                </details>
                {needsTrust ? (
                  <div className="trust-box">
                    <strong>确认此版本的代码来源</strong>
                    <p>
                      版本 <code>{selectedProject.version.slice(0, 16)}</code>
                    </p>
                    <input
                      aria-label="代码来源"
                      placeholder="例如：自己编写，或已审查的仓库提交"
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                    />
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={ack}
                        onChange={(e) => setAck(e.target.checked)}
                      />
                      我信任此版本的 Python 代码，允许在所选 Worker 环境执行
                    </label>
                    <button disabled={!ack || !source.trim()} onClick={confirm}>
                      确认可信版本
                    </button>
                  </div>
                ) : (
                  <p className="good">
                    ✓{" "}
                    {selectedProject.trusted
                      ? "此版本已确认可信"
                      : "规格文件已解析"}
                  </p>
                )}
              </>
            )}
          </section>
          <section className="panel form-panel">
            <div className="section-heading">
              <h2>实验规格</h2>
              <div className="segmented">
                <button
                  className={!advanced ? "selected" : ""}
                  onClick={() => {
                    if (advanced) {
                      try {
                        update(JSON.parse(text));
                        setAdvanced(false);
                      } catch (e) {
                        fail(e);
                      }
                    }
                  }}
                >
                  表单
                </button>
                <button
                  className={advanced ? "selected" : ""}
                  onClick={() => {
                    setText(pretty(spec));
                    setAdvanced(true);
                  }}
                >
                  高级 JSON
                </button>
              </div>
            </div>
            <div className="form-grid">
              <Field label="执行环境">
                <select
                  value={environment}
                  onChange={(e) => setEnvironment(e.target.value)}
                >
                  {meta.environments.map((e) => (
                    <option key={e.id} value={e.id}>
                      {e.label} · {names[e.probe?.status] || "待探测"}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="硬件">
                <select
                  value={env?.device === "cpu" ? "cpu" : gpu}
                  onChange={(e) => setGpu(e.target.value)}
                >
                  {env?.device === "cpu" ? (
                    <option value="cpu">CPU</option>
                  ) : (
                    <>
                      <option value="">选择物理 GPU</option>
                      {meta.devices
                        .filter(
                          (d) =>
                            !env?.gpu_uuids?.length ||
                            env.gpu_uuids.includes(d.uuid),
                        )
                        .map((d) => (
                          <option key={d.uuid} value={d.uuid}>
                            {d.name} · {d.uuid.slice(0, 13)} ·{" "}
                            {d.busy ? "忙" : "空闲"}
                          </option>
                        ))}
                    </>
                  )}
                </select>
              </Field>
            </div>
            {advanced ? (
              <>
                <textarea
                  className="spec-editor mono"
                  aria-label="高级实验 JSON"
                  spellCheck={false}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
                <p className="subtle">
                  使用平台原生 run/check 或 research schema。
                </p>
              </>
            ) : (
              <>
                <div className="form-grid">
                  <Field label="实验名称" wide>
                    <input
                      value={spec.name || ""}
                      onChange={(e) =>
                        update({ ...spec, name: e.target.value })
                      }
                    />
                  </Field>
                  <Field label="Operator">
                    <select
                      value={spec.operator || ""}
                      onChange={(e) => changeOperator(e.target.value)}
                    >
                      {!operators.some((a) => a.ref === spec.operator) && (
                        <option value={spec.operator}>{spec.operator}</option>
                      )}
                      {operators.map((a) => (
                        <option key={a.ref} value={a.ref}>
                          {short(a.ref)}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="dtype">
                    <select
                      value={spec.dtype || "float32"}
                      onChange={(e) =>
                        update({ ...spec, dtype: e.target.value })
                      }
                    >
                      {(
                        operator?.dtypes || [
                          "float16",
                          "bfloat16",
                          "float32",
                          "float64",
                        ]
                      ).map((d: string) => (
                        <option key={d}>{d}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Kernel" wide>
                    <select
                      value={ref(spec.kernels?.[0])}
                      onChange={(e) =>
                        update({
                          ...spec,
                          kernels: [{ asset: e.target.value }],
                        })
                      }
                    >
                      {!kernels.some(
                        (a) => a.ref === ref(spec.kernels?.[0]),
                      ) && (
                        <option value={ref(spec.kernels?.[0])}>
                          {ref(spec.kernels?.[0])}
                        </option>
                      )}
                      {kernels.map((a) => (
                        <option key={a.ref} value={a.ref}>
                          {short(a.ref)}
                        </option>
                      ))}
                    </select>
                  </Field>
                </div>
                <h3>Shape</h3>
                <div className="dimension-grid">
                  {Object.entries(dims).map(([k, v]) => (
                    <Field key={k} label={k}>
                      <input
                        type="number"
                        min="1"
                        value={Number(v)}
                        onChange={(e) =>
                          update({
                            ...spec,
                            shapes: [
                              { ...dims, [k]: +e.target.value },
                              ...spec.shapes.slice(1),
                            ],
                          })
                        }
                      />
                    </Field>
                  ))}
                </div>
                {spec.shapes?.length > 1 && (
                  <p className="subtle">
                    另有 {spec.shapes.length - 1} 个 Shape，可在高级 JSON
                    中编辑。
                  </p>
                )}
                <h3>搜索方法</h3>
                <div className="method-list">
                  {algorithms.map((a) => (
                    <label className="check" key={a.ref}>
                      <input
                        type="checkbox"
                        checked={(spec.algorithms || []).some(
                          (x: any) => ref(x) === a.ref,
                        )}
                        onChange={(e) =>
                          update({
                            ...spec,
                            algorithms: e.target.checked
                              ? [...(spec.algorithms || []), a.ref]
                              : (spec.algorithms || []).filter(
                                  (x: any) => ref(x) !== a.ref,
                                ),
                          })
                        }
                      />
                      {short(a.ref)}
                    </label>
                  ))}
                </div>
                {!algorithms.length && (
                  <p className="subtle">
                    环境资产正在探测。高级 JSON 可直接指定已注册方法。
                  </p>
                )}
                <div className="form-grid">
                  <Field label="Seed">
                    <input
                      type="number"
                      min="0"
                      value={spec.seed ?? 0}
                      onChange={(e) =>
                        update({ ...spec, seed: +e.target.value })
                      }
                    />
                  </Field>
                  <Field label="每方法观测预算">
                    <input
                      type="number"
                      min="1"
                      value={spec.budget?.max_evaluations ?? 4}
                      onChange={(e) =>
                        update({
                          ...spec,
                          budget: {
                            ...spec.budget,
                            max_evaluations: +e.target.value,
                          },
                        })
                      }
                    />
                  </Field>
                  <Field label="每方法墙钟预算（秒）">
                    <input
                      type="number"
                      min="1"
                      value={spec.budget?.max_wall_time_s ?? 30}
                      onChange={(e) =>
                        update({
                          ...spec,
                          budget: {
                            ...spec.budget,
                            max_wall_time_s: +e.target.value,
                          },
                        })
                      }
                    />
                  </Field>
                  <Field label="操作">
                    <select
                      value={action}
                      onChange={(e) => setAction(e.target.value)}
                    >
                      <option value="run">Run · 搜索并验收</option>
                      <option value="check">Check · 独立验收</option>
                    </select>
                  </Field>
                </div>
                <details className="disclosure" open={false}>
                  <summary>
                    实际参数空间 ·{" "}
                    {Object.keys(kernel?.space?.parameters || {}).length} 个参数
                  </summary>
                  {Object.entries(kernel?.space?.parameters || {}).map(
                    ([name, p]: [string, any]) => (
                      <div className="space-domain" key={name}>
                        <b className="mono">{name}</b>
                        <div className="domain-values">
                          {p.values.map((v: any) => (
                            <label className="check" key={JSON.stringify(v)}>
                              <input
                                type="checkbox"
                                checked={
                                  !spec.kernels?.[0]?.space?.[name] ||
                                  spec.kernels[0].space[name].some(
                                    (x: any) =>
                                      JSON.stringify(x) === JSON.stringify(v),
                                  )
                                }
                                onChange={(e) => {
                                  const choice = spec.kernels[0];
                                  const selected =
                                    choice.space?.[name] || p.values;
                                  const next = e.target.checked
                                    ? [...selected, v]
                                    : selected.filter(
                                        (x: any) =>
                                          JSON.stringify(x) !==
                                          JSON.stringify(v),
                                      );
                                  update({
                                    ...spec,
                                    kernels: [
                                      {
                                        ...choice,
                                        space: {
                                          ...(choice.space || {}),
                                          [name]: next,
                                        },
                                      },
                                    ],
                                  });
                                }}
                              />
                              {JSON.stringify(v)}
                            </label>
                          ))}
                        </div>
                        {p.active_when &&
                          Object.keys(p.active_when).length > 0 && (
                            <small>条件：{JSON.stringify(p.active_when)}</small>
                          )}
                      </div>
                    ),
                  )}
                  {!!kernel?.space?.resolver && (
                    <p className="subtle">
                      包含原生动态域与约束；运行时按 Shape、设备和组件身份解析。
                    </p>
                  )}
                </details>
                <details className="disclosure">
                  <summary>正确性、计时与 Operator 参数</summary>
                  <p className="good">独立参考验证 · 容差由 Operator 定义</p>
                  <ParameterTable value={operator?.tolerances} />
                  <div className="form-grid">
                    <Field label="Warmup">
                      <input
                        type="number"
                        min="0"
                        value={spec.warmup ?? 2}
                        onChange={(e) =>
                          update({ ...spec, warmup: +e.target.value })
                        }
                      />
                    </Field>
                    <Field label="Repetitions">
                      <input
                        type="number"
                        min="1"
                        value={spec.repetitions ?? 7}
                        onChange={(e) =>
                          update({ ...spec, repetitions: +e.target.value })
                        }
                      />
                    </Field>
                    <Field label="计时协议">
                      <select
                        value={spec.timing_mode || "full_flow"}
                        onChange={(e) =>
                          update({ ...spec, timing_mode: e.target.value })
                        }
                      >
                        <option value="full_flow">Full flow</option>
                        <option value="cuda_graph">CUDA Graph</option>
                      </select>
                    </Field>
                    <Field label="Graph batch">
                      <input
                        type="number"
                        min="1"
                        value={spec.cuda_graph_batch_size ?? 128}
                        onChange={(e) =>
                          update({
                            ...spec,
                            cuda_graph_batch_size: +e.target.value,
                          })
                        }
                      />
                    </Field>
                    <Field label="独立验收配置预算">
                      <input
                        type="number"
                        min="0"
                        value={spec.validation?.max_configs ?? 1}
                        onChange={(e) =>
                          update({
                            ...spec,
                            validation: {
                              ...spec.validation,
                              max_configs: +e.target.value,
                              trust_claims: false,
                            },
                          })
                        }
                      />
                    </Field>
                    <Field label="验收墙钟（秒）">
                      <input
                        type="number"
                        min="1"
                        value={spec.validation?.max_wall_time_s ?? 30}
                        onChange={(e) =>
                          update({
                            ...spec,
                            validation: {
                              ...spec.validation,
                              max_wall_time_s: +e.target.value,
                              trust_claims: false,
                            },
                          })
                        }
                      />
                    </Field>
                  </div>
                  <Field label="Operator 参数 JSON">
                    <textarea
                      className="mono"
                      rows={4}
                      defaultValue={pretty(spec.parameters || {})}
                      onBlur={(e) => {
                        try {
                          update({
                            ...spec,
                            parameters: JSON.parse(e.target.value),
                          });
                        } catch (err) {
                          fail(err);
                        }
                      }}
                    />
                  </Field>
                </details>
              </>
            )}
          </section>
        </div>
        <aside className="create-summary panel">
          <span className="eyebrow">SUBMISSION</span>
          <h2>运行准备</h2>
          <ParameterTable
            value={{
              环境: env?.label || environment,
              硬件: env?.device === "cpu" ? "CPU" : gpu || "待选择",
              搜索任务: planned ?? "原生预检展开",
              最大观测:
                planned == null
                  ? "原生预检展开"
                  : planned * (spec.budget?.max_evaluations || 0),
              独立验收: spec.validation?.max_configs ?? "—",
            }}
          />
          {needsTrust && <p className="amber">待确认上传代码版本</p>}
          {env?.device === "cuda" && !gpu && (
            <p className="amber">请选择物理 GPU</p>
          )}
          <p className="subtle">预检后确认展开预算与环境，再正式提交。</p>
          <button
            className="primary full"
            disabled={busy || uploadBusy || needsTrust}
            onClick={() => save(true)}
          >
            {busy ? "正在保存…" : "预检并继续 →"}
          </button>
          {notice && (
            <p className="good" role="status">
              {notice}
            </p>
          )}
        </aside>
      </div>
      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <section className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="section-heading">
              <h2>{preview.path}</h2>
              <button onClick={() => setPreview(null)}>关闭</button>
            </div>
            <p className="mono subtle">SHA256 {preview.sha256}</p>
            <pre className="source-preview">{preview.content}</pre>
          </section>
        </div>
      )}
    </>
  );
}

type Trace = {
  key: string;
  name: string;
  context: string;
  label: string;
  rows: Data[];
  stop?: string;
};
function traceData(events: Data[], native: Data): Trace[] {
  const live = new Map<
    string,
    { generation: number; trial: Data; rows: Data[] }
  >();
  for (const event of events) {
    const data = event.data;
    if (event.kind !== "observation" || !data?.observation) continue;
    const key = data.source;
    const generation = data.generation || 0;
    const old = live.get(key);
    if (!old || generation > old.generation)
      live.set(key, {
        generation,
        trial: data.trial,
        rows: [data.observation],
      });
    else if (generation === old.generation) old.rows.push(data.observation);
  }
  const traces = new Map<string, Trace>();
  for (const [key, value] of live) {
    const t = value.trial;
    traces.set(t.trial_id || key, {
      key: t.trial_id || key,
      name: short(t.algorithm_ref),
      context:
        [...new Set(value.rows.map((r) => r.cohort).filter(Boolean))]
          .sort()
          .join(",") ||
        JSON.stringify([
          t.case_id || key,
          t.kernel_fingerprint,
          t.space_fingerprint,
          t.seed,
          t.warmup,
          t.repetitions,
        ]),
      label: `${JSON.stringify(t.shape || {})} · seed ${t.seed ?? "—"}`,
      rows: value.rows,
    });
  }
  for (const detail of native.details || [])
    for (const t of detail.trials || []) {
      if (!t.observation_trace?.length) continue;
      const c = t.case_id || t.cohorts?.[0] || t.trial_id;
      traces.set(t.trial_id, {
        key: t.trial_id,
        name: short(t.algorithm_ref),
        context: t.cohorts?.length
          ? t.cohorts.join(",")
          : JSON.stringify([c, t.seed, t.warmup, t.repetitions]),
        label: JSON.stringify(t.shape || {}),
        rows: t.observation_trace,
        stop: t.stop_reason,
      });
    }
  return [...traces.values()];
}

function SearchChart({
  traces,
  title = "搜索过程",
}: {
  traces: Trace[];
  title?: string;
}) {
  const [axis, setAxis] = useState("observation");
  const [context, setContext] = useState("");
  const contexts = [
    ...new Map(traces.map((t) => [t.context, t.label])).entries(),
  ];
  const active = contexts.some(([key]) => key === context)
    ? context
    : contexts[0]?.[0];
  const selected = traces.filter((t) => t.context === active);
  const option = useMemo(
    () => ({
      ...baseChart,
      xAxis: {
        type: "value",
        name: axis === "observation" ? "观测次数" : "实际墙钟 / s",
        min: 0,
        nameLocation: "middle",
        nameGap: 30,
        minInterval: axis === "observation" ? 1 : undefined,
        axisLine: { lineStyle: { color: "#c5d0df" } },
        axisLabel: { color: "#61738c" },
      },
      yAxis: {
        type: "value",
        name: "Best verified / ms",
        scale: true,
        splitLine: { lineStyle: { color: "#edf1f6" } },
        axisLabel: { formatter: (v: number) => num(v) },
      },
      series: selected.map((t) => ({
        name: t.name,
        type: "line",
        step: "end",
        showSymbol: t.rows.length <= 200,
        symbolSize: 6,
        connectNulls: false,
        emphasis: { focus: "series" },
        lineStyle: { width: 2 },
        data: t.rows.map((r) => ({
          value: [
            axis === "observation" ? r.observation : r.elapsed_wall_time_s,
            r.best_verified_latency_ms,
          ],
          symbol: r.cache_hit
            ? "emptyDiamond"
            : r.status === "success"
              ? "circle"
              : "rect",
          name: `${r.cache_hit ? "缓存" : "测量"} · ${names[r.status] || r.status} · ${String(r.config_id).slice(0, 10)}`,
        })),
      })),
    }),
    [
      selected
        .map(
          (t) =>
            `${t.key}:${t.rows.length}:${t.rows.at(-1)?.elapsed_wall_time_s}`,
        )
        .join("|"),
      axis,
    ],
  );
  return (
    <section className="panel curve-panel">
      <div className="section-heading">
        <div>
          <h2>{title}</h2>
          <span className="subtle">Best-so-far · 已通过参考验证的延迟</span>
        </div>
        <div className="segmented">
          <button
            className={axis === "observation" ? "selected" : ""}
            onClick={() => setAxis("observation")}
          >
            观测次数
          </button>
          <button
            className={axis === "wall" ? "selected" : ""}
            onClick={() => setAxis("wall")}
          >
            实际墙钟
          </button>
        </div>
      </div>
      {contexts.length > 1 && (
        <div className="context-select">
          <label>
            测量上下文{" "}
            <select
              aria-label="测量上下文"
              value={active}
              onChange={(e) => setContext(e.target.value)}
            >
              {contexts.map(([key, label]) => (
                <option key={key} value={key}>
                  {label} · {key.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      {selected.some((t) =>
        t.rows.some(
          (r) =>
            r.best_verified_latency_ms != null &&
            (axis === "observation" || r.elapsed_wall_time_s != null),
        ),
      ) ? (
        <Chart option={option} label="搜索过程" height={370} />
      ) : (
        <Empty>{axis === "wall" ? "尚无逐步墙钟记录" : "尚无逐观测轨迹"}</Empty>
      )}
      {selected.some((t) => t.stop) && (
        <div className="chart-stops subtle">
          {selected
            .filter((t) => t.stop)
            .map((t) => (
              <span key={t.key}>
                {t.name} · {t.stop}
              </span>
            ))}
        </div>
      )}
    </section>
  );
}

function JobDetail({
  id,
  meta,
  refresh,
  navigate,
  fail,
}: {
  id: string;
  meta: Meta;
  refresh: () => Promise<void>;
  navigate: (x: string) => void;
  fail: (e: any) => void;
}) {
  const [job, setJob] = useState<Data | null>(null);
  const [native, setNative] = useState<Data>({ details: [] });
  const [events, setEvents] = useState<Data[]>([]);
  const [tab, setTab] = useState("search");
  const [busy, setBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const [seq, setSeq] = useState(0);
  const [phase, setPhase] = useState<Data>({});
  const queue = useRef<Data[]>([]);
  const lastSeq = useRef(0);
  const nextRefresh = useRef<ReturnType<typeof setTimeout> | null>(null);
  async function reload() {
    const j = await api("/api/jobs/" + id);
    setJob(j);
    if (["completed", "failed", "interrupted", "cancelled"].includes(j.state)) {
      setNative(await api(`/api/jobs/${id}/results`));
    }
    return j;
  }
  useEffect(() => {
    setJob(null);
    setEvents([]);
    setNative({ details: [] });
    setSeq(0);
    lastSeq.current = 0;
    queue.current = [];
    reload().catch(fail);
    const source = new EventSource(`/api/jobs/${id}/events`);
    source.onopen = () => {
      setConnected(true);
      reload().catch((e) => {
        if (!(e instanceof TypeError)) fail(e);
      });
    };
    source.onerror = () => setConnected(false);
    source.onmessage = (message) => {
      const e = JSON.parse(message.data);
      if (e.seq <= lastSeq.current) return;
      lastSeq.current = e.seq;
      if (e.kind === "observation") queue.current.push(e);
      if (e.kind === "phase") setPhase(e.data);
      if (e.kind === "state" && !nextRefresh.current)
        nextRefresh.current = setTimeout(() => {
          nextRefresh.current = null;
          reload().catch(fail);
          refresh().catch(fail);
        }, 200);
    };
    const paint = setInterval(() => {
      setSeq(lastSeq.current);
      if (queue.current.length) {
        const batch = queue.current;
        queue.current = [];
        setEvents((old) => [...old, ...batch]);
      }
    }, 250);
    return () => {
      source.close();
      clearInterval(paint);
      if (nextRefresh.current) clearTimeout(nextRefresh.current);
      nextRefresh.current = null;
    };
  }, [id]);
  async function operation(op: string) {
    if (!job) return;
    setBusy(true);
    try {
      const body =
        op === "submit"
          ? {
              plan_sha: job.plan_sha,
              idempotency_key: `${job.id}:${job.plan_sha}`,
            }
          : {};
      const result = await api(`/api/jobs/${id}/${op}`, body);
      if (op === "duplicate") {
        navigate("new/" + result.id);
      } else {
        if (op === "resume") setNative({ details: [] });
        await reload();
        await refresh();
      }
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  if (!job) return <Empty>加载实验…</Empty>;
  const traces = traceData(events, native);
  const observed = traces.reduce((n, t) => n + t.rows.length, 0);
  const cached = traces.reduce(
    (n, t) => n + t.rows.filter((r) => r.cache_hit).length,
    0,
  );
  const verified = traces.reduce(
    (n, t) => n + t.rows.filter((r) => r.correctness_verified).length,
    0,
  );
  const failures = traces.reduce(
    (n, t) => n + t.rows.filter((r) => r.status !== "success").length,
    0,
  );
  const best = traces
    .flatMap((t) => t.rows.map((r) => r.best_verified_latency_ms))
    .filter((v) => v != null);
  const env = meta.environments.find((e) => e.id === job.environment);
  return (
    <>
      <a className="back-link" href="#workspace">
        ← 实验工作区
      </a>
      <div className="page-heading job-heading">
        <div>
          <div className="eyebrow">EXPERIMENT / {id.slice(0, 8)}</div>
          <h1>
            {job.name} <Status value={job.state} />
          </h1>
          <p>
            {env?.label || job.environment} · {short(job.spec?.operator)} ·{" "}
            {job.spec?.dtype} · seed {job.spec?.seed}
          </p>
        </div>
        <div className="actions">
          <button onClick={() => operation("duplicate")} disabled={busy}>
            复用配置
          </button>
          {job.state === "draft" && (
            <button onClick={() => navigate("new/" + id)}>编辑草稿</button>
          )}
          {job.state === "ready" && (
            <button
              className="primary"
              disabled={busy}
              onClick={() => operation("submit")}
            >
              ▶ 提交实验
            </button>
          )}
          {["queued", "running", "prechecking", "cancelling"].includes(
            job.state,
          ) && (
            <button
              className="danger"
              disabled={busy || job.state === "cancelling"}
              onClick={() => operation("cancel")}
            >
              取消任务
            </button>
          )}
          {(["failed", "interrupted", "cancelled"].includes(job.state) ||
            job.resource_blocked) &&
            job.precheck && (
              <button
                className="primary"
                disabled={busy}
                onClick={() => operation("resume")}
              >
                {job.resource_blocked ? "重试资源准入" : "继续实验"}
              </button>
            )}
        </div>
      </div>
      {job.error && (
        <div
          className={job.resource_blocked ? "notice-banner" : "error-banner"}
          role="status"
        >
          {typeof job.error === "object" ? pretty(job.error) : job.error}
        </div>
      )}
      {["ready", "prechecking", "invalid", "draft"].includes(job.state) && (
        <section className="panel precheck-panel">
          <div className="section-heading">
            <h2>提交前预检</h2>
            <Status value={job.precheck?.status || job.state} />
          </div>
          <div className="budget-grid">
            <div>
              <strong>{job.planned?.execution_jobs ?? "—"}</strong>
              <span>展开执行任务</span>
            </div>
            <div>
              <strong>{job.planned?.search_trials ?? "—"}</strong>
              <span>搜索 Trial</span>
            </div>
            <div>
              <strong>{job.planned?.max_observations ?? "—"}</strong>
              <span>最大算法观测</span>
            </div>
            <div>
              <strong>{job.planned?.max_validation_evaluations ?? "—"}</strong>
              <span>独立验收预算</span>
            </div>
            <div>
              <strong>{num(job.planned?.serial_wall_allowance_s, " s")}</strong>
              <span>串行墙钟上限</span>
            </div>
          </div>
          {job.precheck?.warnings?.map((w: string) => (
            <p className="amber" key={w}>
              {w}
            </p>
          ))}
          {job.precheck?.effective_spaces && (
            <details className="disclosure">
              <summary>已解析资产与参数空间</summary>
              {job.precheck.effective_spaces.map((s: Data, i: number) => (
                <div key={i}>
                  <h3>{short(s.kernel)}</h3>
                  <ParameterTable
                    value={Object.fromEntries(
                      Object.entries(s.declared.parameters).map(
                        ([k, v]: [string, any]) => [k, v.values],
                      ),
                    )}
                  />
                </div>
              ))}
            </details>
          )}
        </section>
      )}
      <div className="run-strip">
        <div>
          <i className={connected ? "dot live" : "dot"} />
          <span>
            {connected ? "事件已连接" : "正在重连"}
            <small>
              #{seq} · {phase.phase || job.last_phase?.phase || "等待执行"}
            </small>
          </span>
        </div>
        <div>
          <b>{observed}</b>
          <span>观测 / {job.planned?.max_observations ?? "—"}</span>
        </div>
        <div>
          <b>{verified}</b>
          <span>已验证</span>
        </div>
        <div>
          <b>{cached}</b>
          <span>缓存</span>
        </div>
        <div>
          <b className={failures ? "red" : ""}>{failures}</b>
          <span>失败</span>
        </div>
        <div>
          <b>{best.length ? num(Math.min(...best), " ms") : "—"}</b>
          <span>Best verified</span>
        </div>
      </div>
      <div className="tabs" role="tablist">
        {[
          ["search", "搜索过程"],
          ["configs", "配置结果"],
          ["cost", "成本"],
          ["files", "文件与下载"],
          ["logs", "运行日志"],
          ["identity", "完整身份"],
        ].map(([key, label]) => (
          <button
            role="tab"
            aria-selected={tab === key}
            key={key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "search" ? (
        <>
          <SearchChart traces={traces} />
          {(phase.config || phase.progress?.config) && (
            <section className="panel form-panel">
              <h2>当前候选</h2>
              <ParameterTable value={phase.config || phase.progress?.config} />
            </section>
          )}
          <div className="two-columns">
            <Exploration native={native} traces={traces} />
            <section className="panel form-panel">
              <h2>最近观测</h2>
              {traces
                .flatMap((t) => t.rows.slice(-2).map((r: Data) => ({ t, r })))
                .slice(-6)
                .reverse()
                .map(({ t, r }, i) => (
                  <div className="recent-row" key={i}>
                    <div>
                      <strong>
                        {t.name} · #{r.observation}
                      </strong>
                      <small className="mono">
                        {r.config_id?.slice(0, 14)}
                      </small>
                    </div>
                    <div>
                      <Status value={r.status} />
                      <small>
                        {r.cache_hit ? "缓存复用" : "真实测量"} ·{" "}
                        {num(r.latency_ms, " ms")}
                      </small>
                    </div>
                  </div>
                ))}
              {!observed && <Empty>首条观测完成后显示结果</Empty>}
            </section>
          </div>
        </>
      ) : tab === "configs" ? (
        <ConfigurationTable
          native={native}
          traces={traces}
          jobId={id}
          fail={fail}
        />
      ) : tab === "cost" ? (
        <Costs traces={traces} native={native} />
      ) : tab === "files" ? (
        <Files jobId={id} taskState={job.state} fail={fail} />
      ) : tab === "logs" ? (
        <Logs jobId={id} fail={fail} />
      ) : (
        <section className="panel form-panel">
          <h2>冻结身份</h2>
          <ParameterTable
            value={{
              任务: id,
              规格SHA: job.plan?.spec_sha256,
              平台SHA: job.plan?.platform_sha256,
              环境SHA: job.plan?.environment_sha256,
              代码版本: job.plan?.project_version || "服务器资产",
              GPU: job.gpu || "CPU",
            }}
          />
          <details className="disclosure" open>
            <summary>原生实验规格</summary>
            <Json value={job.spec} />
          </details>
          <details className="disclosure">
            <summary>环境、资源与预检身份</summary>
            <Json
              value={{
                plan: job.plan,
                precheck: job.precheck,
                execution: job.execution,
                last_result: job.last_result,
              }}
            />
          </details>
        </section>
      )}
    </>
  );
}

function allRecords(native: Data): Data[] {
  return (native.details || []).flatMap((d: Data) =>
    d.records.map((r: Data) => ({ ...r, dataset: d.id })),
  );
}
function liveRecords(traces: Trace[]): Data[] {
  return [
    ...new Map(
      traces.flatMap((t) =>
        t.rows
          .filter((r) => r.config && r.dataset)
          .map(
            (r) =>
              [
                `${r.dataset}:${r.record_id}`,
                {
                  id: r.record_id,
                  dataset: r.dataset,
                  config_id: r.config_id,
                  params: r.config,
                  status: r.status,
                  verified: r.correctness_verified,
                  latency_ms: r.latency_ms,
                  dtype: r.dtype,
                  implementation: r.implementation,
                  samples: r.samples_count,
                },
              ] as [string, Data],
          ),
      ),
    ).values(),
  ];
}
function Exploration({
  native,
  traces = [],
}: {
  native: Data;
  traces?: Trace[];
}) {
  const records = allRecords(native).length
    ? allRecords(native)
    : liveRecords(traces);
  const parameters = [
    ...new Set(records.flatMap((r) => Object.keys(r.params || {}))),
  ];
  const [parameter, setParameter] = useState("");
  const active = parameters.includes(parameter) ? parameter : parameters[0];
  const counts = new Map<string, number>();
  for (const r of records) {
    const value = JSON.stringify(r.params?.[active]);
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  const option = {
    ...baseChart,
    legend: { show: false },
    grid: { left: 55, right: 22, top: 40, bottom: 70 },
    xAxis: { type: "category", data: [...counts.keys()], name: active },
    yAxis: { type: "value", name: "记录数", minInterval: 1 },
    series: [{ type: "bar", barMaxWidth: 35, data: [...counts.values()] }],
  };
  return (
    <section className="panel form-panel">
      <div className="section-heading">
        <h2>参数探索</h2>
        {parameters.length > 0 && (
          <select
            aria-label="探索参数"
            value={active}
            onChange={(e) => setParameter(e.target.value)}
          >
            {parameters.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
        )}
      </div>
      {records.length ? (
        <Chart option={option} label="参数探索" height={260} />
      ) : (
        <Empty>原生配置记录生成后显示分布</Empty>
      )}
    </section>
  );
}

function ConfigurationTable({
  native,
  traces = [],
  jobId,
  fail,
}: {
  native: Data;
  traces?: Trace[];
  jobId?: string;
  fail: (e: any) => void;
}) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState(0);
  const [detail, setDetail] = useState<Data | null>(null);
  const records = allRecords(native).length
    ? allRecords(native)
    : liveRecords(traces);
  const rows = records.filter(
    (r) =>
      (!status || r.status === status) &&
      `${r.config_id} ${JSON.stringify(r.params)}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const visible = rows.slice(page * 25, (page + 1) * 25);
  async function select(r: Data) {
    if (!jobId) {
      setDetail(r);
      return;
    }
    try {
      setDetail(await api(`/api/jobs/${jobId}/records/${r.dataset}/${r.id}`));
    } catch (e) {
      fail(e);
    }
  }
  return (
    <section className="panel form-panel">
      <div className="section-heading">
        <h2>配置结果</h2>
        <div className="actions">
          {jobId && (
            <>
              <a
                className="button"
                href={`/api/jobs/${jobId}/table?format=csv`}
              >
                下载 CSV
              </a>
              <a
                className="button"
                href={`/api/jobs/${jobId}/table?format=json`}
              >
                下载 JSON
              </a>
            </>
          )}
        </div>
      </div>
      <div className="toolbar">
        <input
          aria-label="筛选配置"
          value={search}
          placeholder="搜索配置 ID 或参数"
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
        />
        <select
          aria-label="配置状态"
          value={status}
          onChange={(e) => {
            setStatus(e.target.value);
            setPage(0);
          }}
        >
          <option value="">全部状态</option>
          {[...new Set(records.map((r) => r.status))].map((s) => (
            <option key={s} value={s}>
              {names[s] || s}
            </option>
          ))}
        </select>
      </div>
      {rows.length ? (
        <>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>配置</th>
                  <th>参数</th>
                  <th>延迟 / ms</th>
                  <th>状态</th>
                  <th>正确性</th>
                  <th>样本</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r, i) => (
                  <tr key={r.id + ":" + i}>
                    <td>
                      <button
                        className="row-link mono"
                        onClick={() => select(r)}
                      >
                        {r.config_id.slice(0, 12)}
                      </button>
                      <small>
                        {r.dtype} · {short(r.implementation)}
                      </small>
                    </td>
                    <td className="param-cell">
                      {Object.entries(r.params || {}).map(([k, v]) => (
                        <span key={k}>
                          <b>{k}</b>={JSON.stringify(v)}
                        </span>
                      ))}
                    </td>
                    <td className="numeric">{num(r.latency_ms)}</td>
                    <td>
                      <Status value={r.status} />
                    </td>
                    <td>
                      {r.verified ? (
                        <span className="good">参考验证通过</span>
                      ) : (
                        <span className="amber">未验证</span>
                      )}
                    </td>
                    <td>{r.samples}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination">
            <span>{rows.length} 条原生记录</span>
            <button disabled={page === 0} onClick={() => setPage((x) => x - 1)}>
              上一页
            </button>
            <span>{page + 1}</span>
            <button
              disabled={(page + 1) * 25 >= rows.length}
              onClick={() => setPage((x) => x + 1)}
            >
              下一页
            </button>
          </div>
        </>
      ) : (
        <Empty>没有匹配的配置记录</Empty>
      )}
      {detail && (
        <div className="modal-backdrop" onClick={() => setDetail(null)}>
          <section className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="section-heading">
              <h2>配置详情</h2>
              <button onClick={() => setDetail(null)}>关闭</button>
            </div>
            <h3>参数</h3>
            <ParameterTable value={detail.config?.params || detail.params} />
            <h3>正确性与执行</h3>
            <Json
              value={
                detail.metadata || {
                  status: detail.status,
                  verified: detail.verified,
                  failure: detail.failure,
                }
              }
            />
            <details className="disclosure">
              <summary>完整原生记录</summary>
              <Json value={detail} />
            </details>
          </section>
        </div>
      )}
    </section>
  );
}

function Costs({ traces, native }: { traces: Trace[]; native: Data }) {
  const totals: Data = {};
  const physical: Data = {};
  for (const t of traces) {
    const last = t.rows.at(-1);
    for (const [key, value] of Object.entries(last?.algorithm_costs_ms || {}))
      totals[key] = (totals[key] || 0) + Number(value);
    for (const r of t.rows) {
      if (r.physical_compile_time_ms != null)
        physical.compile_ms =
          (physical.compile_ms || 0) + Number(r.physical_compile_time_ms);
      if (r.physical_measurement_wall_time_s != null)
        physical.measurement_ms =
          (physical.measurement_ms || 0) +
          Number(r.physical_measurement_wall_time_s) * 1000;
    }
  }
  const costs = { ...totals, ...physical };
  const labels: Data = {
    factory_ms: "搜索器构造",
    proposal_ms: "候选生成",
    feedback_ms: "结果反馈",
    shutdown_ms: "搜索器收尾",
    compile_ms: "实际编译",
    measurement_ms: "查找与测量",
  };
  const option = {
    ...baseChart,
    legend: { show: false },
    dataZoom: [],
    grid: { left: 125, right: 45, top: 35, bottom: 40 },
    xAxis: { type: "value", name: "ms" },
    yAxis: {
      type: "category",
      data: Object.keys(costs).map((k) => labels[k] || k),
    },
    series: [{ type: "bar", barMaxWidth: 26, data: Object.values(costs) }],
  };
  const diagnostics = traces
    .map((t) => ({
      algorithm: t.name,
      diagnostics: t.rows.at(-1)?.algorithm_diagnostics,
    }))
    .filter((t) => t.diagnostics && Object.keys(t.diagnostics).length);
  return (
    <section className="panel form-panel">
      <div className="section-heading">
        <h2>已记录成本</h2>
        <span className="subtle">各项独立统计 · 缓存复用不计物理测量成本</span>
      </div>
      {traces.length ? (
        <Chart option={option} label="成本分析" height={320} />
      ) : (
        <Empty>暂无成本记录</Empty>
      )}
      <div className="cost-counters">
        <div>
          <b>
            {traces.reduce(
              (n, t) => n + t.rows.filter((r) => !r.cache_hit).length,
              0,
            )}
          </b>{" "}
          真实尝试
        </div>
        <div>
          <b>
            {traces.reduce(
              (n, t) => n + t.rows.filter((r) => r.cache_hit).length,
              0,
            )}
          </b>{" "}
          缓存揭示
        </div>
      </div>
      <details className="disclosure">
        <summary>空间构造、模型 / API 诊断</summary>
        {diagnostics.length ? (
          diagnostics.map((d, i) => (
            <div key={i}>
              <h3>{d.algorithm}</h3>
              <Json value={d.diagnostics} />
            </div>
          ))
        ) : (
          <p className="subtle">此实验未保存额外诊断字段。</p>
        )}
      </details>
    </section>
  );
}

function Files({
  jobId,
  taskState,
  fail,
}: {
  jobId: string;
  taskState: string;
  fail: (e: any) => void;
}) {
  const [files, setFiles] = useState<Data[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const [result, setResult] = useState<Data | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    api(`/api/jobs/${jobId}/files`)
      .then((x) => {
        setFiles(x.files);
        setSelected(
          x.files
            .filter(
              (f: Data) =>
                f.path === "output/report.md" ||
                f.path === "output/run_summary.json" ||
                f.path === "output/experiment.json",
            )
            .map((f: Data) => f.path),
        );
      })
      .catch(fail);
  }, [jobId]);
  async function download() {
    setBusy(true);
    try {
      const x = await api(`/api/jobs/${jobId}/exports`, { files: selected });
      setResult(x);
      const a = document.createElement("a");
      a.href = x.download;
      a.download = "results.zip";
      a.click();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  const visible = files.filter((f) => !filter || f.kind === filter);
  return (
    <section className="panel form-panel">
      <div className="section-heading">
        <div>
          <h2>选择性下载</h2>
          <span className="subtle">
            只打包所选文件 · 自动附带文件清单和 SHA256
          </span>
        </div>
        <button
          className="primary"
          disabled={busy || !selected.length}
          onClick={download}
        >
          {busy ? "正在打包…" : `下载所选 ${selected.length} 个文件`}
        </button>
      </div>
      <div className="toolbar">
        <select
          value={filter}
          aria-label="文件类型"
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="">全部文件</option>
          <option value="report">报告 / 图表</option>
          <option value="configuration">配置 / JSON</option>
          <option value="database">SQLite</option>
          <option value="log">日志 / JSONL</option>
          <option value="source">冻结源码</option>
          <option value="compiled">编译产物</option>
        </select>
        <button
          onClick={() =>
            setSelected([
              ...new Set([...selected, ...visible.map((f) => f.path)]),
            ])
          }
        >
          选择当前列表
        </button>
        <button onClick={() => setSelected([])}>清空选择</button>
        <span className="subtle">
          已选{" "}
          {bytes(
            files
              .filter((f) => selected.includes(f.path))
              .reduce((n, f) => n + f.bytes, 0),
          )}
        </span>
      </div>
      <div className="download-list">
        {visible.map((f) => (
          <label className="download-row" key={f.path}>
            <input
              type="checkbox"
              checked={selected.includes(f.path)}
              onChange={(e) =>
                setSelected((old) =>
                  e.target.checked
                    ? [...old, f.path]
                    : old.filter((x) => x !== f.path),
                )
              }
            />
            <span className="file-path">{f.path}</span>
            <span className="subtle nowrap">{bytes(f.bytes)}</span>
            {!["running", "queued", "prechecking", "cancelling"].includes(
              taskState,
            ) && f.kind !== "database" ? (
              <a
                className="text-button"
                aria-label={`下载 ${f.path}`}
                href={`/api/jobs/${jobId}/download?path=${encodeURIComponent(f.path)}`}
                download
              >
                下载
              </a>
            ) : (
              <span />
            )}
          </label>
        ))}
      </div>
      {!files.length && <Empty>尚未生成实验文件</Empty>}
      {result && (
        <div className="export-receipt">
          <span className="good">✓ 导出完成 · {bytes(result.bytes)}</span>
          <code>SHA256 {result.sha256}</code>
          <a className="text-button" href={result.download}>
            再次下载 ↗
          </a>
        </div>
      )}
    </section>
  );
}

function Logs({ jobId, fail }: { jobId: string; fail: (e: any) => void }) {
  const [logs, setLogs] = useState<Data[]>([]);
  useEffect(() => {
    const load = () =>
      api(`/api/jobs/${jobId}/logs`)
        .then((x) => setLogs(x.logs))
        .catch(fail);
    load();
    const id = setInterval(load, 2500);
    return () => clearInterval(id);
  }, [jobId]);
  return (
    <section className="panel form-panel">
      <div className="section-heading">
        <h2>运行日志</h2>
        <span className="subtle">最近 64 KB / 文件</span>
      </div>
      {logs.map((x, i) => (
        <div key={i}>
          <h3>{x.file}</h3>
          <pre className="log-view">{x.content || "暂无日志输出"}</pre>
        </div>
      ))}
      {!logs.length && <Empty>Worker 尚未生成日志</Empty>}
    </section>
  );
}

function Assets({ meta }: { meta: Meta }) {
  const [environment, setEnvironment] = useState("core");
  const [kind, setKind] = useState("kernel");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Data | null>(null);
  const env = meta.environments.find((e) => e.id === environment);
  const rows = (env?.probe?.assets || []).filter(
    (a: Data) =>
      a.kind === kind && a.ref.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">REGISTERED ASSETS</div>
          <h1>算子与算法</h1>
          <p>来自所选服务器环境的实际注册结果。</p>
        </div>
        <a className="button primary" href="#new">
          ＋ 新建实验
        </a>
      </div>
      <div className="toolbar">
        <select
          aria-label="资产环境"
          value={environment}
          onChange={(e) => setEnvironment(e.target.value)}
        >
          {meta.environments.map((e) => (
            <option key={e.id} value={e.id}>
              {e.label}
            </option>
          ))}
        </select>
        <div className="segmented">
          {[
            ["kernel", "Kernel"],
            ["operator", "Operator"],
            ["algorithm", "Searcher"],
          ].map(([k, v]) => (
            <button
              className={kind === k ? "selected" : ""}
              key={k}
              onClick={() => setKind(k)}
            >
              {v}
            </button>
          ))}
        </div>
        <input
          placeholder="搜索资产"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <section className="panel">
        <table>
          <thead>
            <tr>
              <th>资产</th>
              <th>执行路径</th>
              <th>dtype / 空间</th>
              <th>版本</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a: Data) => (
              <tr key={a.ref}>
                <td>
                  <button className="row-link" onClick={() => setSelected(a)}>
                    {short(a.ref)}
                  </button>
                  <small>{short(a.operator || a.kind)}</small>
                </td>
                <td>
                  {a.runtime || "SDK"}
                  <small>{(a.architectures || []).join(" / ")}</small>
                </td>
                <td>
                  {a.dtypes?.join(" / ") ||
                    Object.keys(a.space?.parameters || {}).join(", ") ||
                    "—"}
                </td>
                <td>{a.version}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <Empty>此环境暂未发现匹配资产</Empty>}
      </section>
      {selected && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <section className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="section-heading">
              <h2>{short(selected.ref)}</h2>
              <button onClick={() => setSelected(null)}>关闭</button>
            </div>
            {selected.space && (
              <ParameterTable
                value={Object.fromEntries(
                  Object.entries(selected.space.parameters).map(
                    ([k, v]: [string, any]) => [k, v.values],
                  ),
                )}
              />
            )}
            <Json value={selected} />
          </section>
        </div>
      )}
    </>
  );
}

function Devices({
  meta,
  refresh,
  setMeta,
  fail,
}: {
  meta: Meta;
  refresh: () => Promise<void>;
  setMeta: React.Dispatch<React.SetStateAction<Meta>>;
  fail: (e: any) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState("");
  async function probe() {
    setBusy(true);
    try {
      const x = await api("/api/devices");
      setMeta((m) => ({ ...m, devices: x.devices }));
      await api("/api/environments/probe", {});
      await refresh();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">SERVER RUNTIME</div>
          <h1>设备与环境</h1>
          <p>环境由服务器配置登记，实验使用明确的 Python 与依赖路径。</p>
        </div>
        <button onClick={probe} disabled={busy}>
          {busy ? "正在查询…" : "刷新探测"}
        </button>
      </div>
      <section className="panel form-panel">
        <h2>物理 GPU</h2>
        {meta.devices.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>设备</th>
                  <th>状态</th>
                  <th>显存</th>
                  <th>利用率</th>
                  <th>进程</th>
                </tr>
              </thead>
              <tbody>
                {meta.devices.map((d) => (
                  <tr key={d.uuid}>
                    <td>
                      <strong>{d.name}</strong>
                      <small className="mono">{d.uuid}</small>
                    </td>
                    <td>
                      <Status value={d.busy ? "queued" : "available"} />
                      <small>{d.busy ? "资源忙" : "当前空闲"}</small>
                    </td>
                    <td>
                      {num(d.used_mib)} / {num(d.memory_mib)} MiB
                    </td>
                    <td>{num(d.util_pct)}%</td>
                    <td>{d.process_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>当前服务器未发现可查询的 GPU。CPU 实验可以正常使用。</Empty>
        )}
      </section>
      <section className="panel form-panel">
        <h2>已注册环境</h2>
        {meta.environments.map((env) => (
          <div className="environment" key={env.id}>
            <div className="section-heading">
              <div>
                <h3>{env.label}</h3>
                <span className="subtle">
                  {env.profile} · {env.device.toUpperCase()}
                </span>
              </div>
              <Status value={env.probe?.status || "probing"} />
            </div>
            <ParameterTable
              value={
                env.probe?.runtime?.packages || { 状态: "等待 Worker 探测" }
              }
            />
            <button
              className="text-button"
              onClick={() => setExpanded(expanded === env.id ? "" : env.id)}
            >
              {expanded === env.id ? "收起身份" : "查看完整路径与身份"} ↗
            </button>
            {expanded === env.id && (
              <Json
                value={{
                  python: env.python,
                  env: env.env,
                  runtime: env.probe?.runtime,
                  error: env.probe?.error,
                  read_paths: env.read_paths,
                }}
              />
            )}
          </div>
        ))}
      </section>
    </>
  );
}

function Analysis({ meta, fail }: { meta: Meta; fail: (e: any) => void }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [loaded, setLoaded] = useState<Trace[]>([]);
  const [busy, setBusy] = useState(false);
  async function compare() {
    setBusy(true);
    try {
      const runs = await Promise.all(
        selected.map(async (id) => ({
          id,
          result: await api(`/api/jobs/${id}/results`),
        })),
      );
      const traces = runs.flatMap(({ id, result }) =>
        traceData([], result).map((t) => ({
          ...t,
          key: id + ":" + t.key,
          name: `${meta.jobs.find((j) => j.id === id)?.name} / ${t.name}`,
        })),
      );
      setLoaded(traces);
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RESULTS & COMPARISONS</div>
          <h1>结果分析</h1>
          <p>选择已有实验；不同测量上下文分开展示。</p>
        </div>
        <button
          className="primary"
          disabled={busy || !selected.length}
          onClick={compare}
        >
          {busy ? "正在读取…" : "加载比较"}
        </button>
      </div>
      <section className="panel form-panel">
        <div className="comparison-select">
          {meta.jobs
            .filter((j) => j.last_result)
            .map((j) => (
              <label className="check" key={j.id}>
                <input
                  type="checkbox"
                  checked={selected.includes(j.id)}
                  onChange={(e) =>
                    setSelected((old) =>
                      e.target.checked
                        ? [...old, j.id]
                        : old.filter((x) => x !== j.id),
                    )
                  }
                />
                {j.name}{" "}
                <small>
                  {j.spec?.dtype} · {j.environment}
                </small>
              </label>
            ))}
        </div>
        {!meta.jobs.some((j) => j.last_result) && (
          <Empty>完成实验后可在这里对比搜索过程</Empty>
        )}
      </section>
      <SearchChart traces={loaded} title="多方法搜索曲线" />
    </>
  );
}

function History({ id, fail }: { id?: string; fail: (e: any) => void }) {
  const [rows, setRows] = useState<Data[]>([]);
  const [detail, setDetail] = useState<Data | null>(null);
  const [tab, setTab] = useState("search");
  useEffect(() => {
    if (id)
      api("/api/history/" + id)
        .then(setDetail)
        .catch(fail);
    else
      api("/api/history")
        .then((x) => setRows(x.experiments))
        .catch(fail);
  }, [id]);
  if (!id)
    return (
      <>
        <div className="page-heading">
          <div>
            <div className="eyebrow">EXISTING RESULTS</div>
            <h1>历史实验</h1>
            <p>读取服务器管理员登记的结果目录。</p>
          </div>
        </div>
        <section className="panel form-panel">
          {rows.map((r) => (
            <div className="recent-row" key={r.id}>
              <a className="row-link" href={"#history/" + r.id}>
                {r.experiment || r.label.split("/").at(-1)}
              </a>
              <Status value={r.status} />
            </div>
          ))}
          {!rows.length && (
            <Empty>
              未登记历史目录。在 server.json 的 history_roots 中添加结果路径。
            </Empty>
          )}
        </section>
      </>
    );
  if (!detail) return <Empty>加载历史结果…</Empty>;
  const native = { details: [detail] };
  return (
    <>
      <a className="back-link" href="#history">
        ← 历史实验
      </a>
      <div className="page-heading">
        <div>
          <h1>
            {detail.summary?.experiment || detail.label?.split("/").at(-1)}
          </h1>
          <p>已有实验记录</p>
        </div>
        <Status value={detail.summary?.status || "available"} />
      </div>
      <div className="tabs">
        {[
          ["search", "搜索过程"],
          ["configs", "配置结果"],
          ["cost", "成本"],
          ["l3", "L3 配置集合"],
        ].map(([k, v]) => (
          <button
            className={tab === k ? "active" : ""}
            key={k}
            onClick={() => setTab(k)}
          >
            {v}
          </button>
        ))}
      </div>
      {tab === "search" ? (
        <SearchChart traces={traceData([], native)} />
      ) : tab === "configs" ? (
        <ConfigurationTable native={native} fail={fail} />
      ) : tab === "cost" ? (
        <Costs traces={traceData([], native)} native={native} />
      ) : (
        <L3 value={detail.l3} />
      )}
    </>
  );
}
function L3({ value }: { value: Data }) {
  const rows = value?.results || [];
  const option = {
    ...baseChart,
    legend: { show: false },
    xAxis: { type: "value", name: "K", minInterval: 1 },
    yAxis: { type: "value", name: "Loss" },
    series: [
      {
        type: "line",
        data: rows
          .filter(
            (r: Data) =>
              r.k != null && (r.relative_loss != null || r.loss != null),
          )
          .map((r: Data) => [r.k, r.relative_loss ?? r.loss]),
      },
    ],
  };
  return (
    <section className="panel form-panel">
      <h2>有限配置集合与分派</h2>
      {rows.length ? (
        <>
          <Chart option={option} label="K-Loss" />
          {rows.map((r: Data, i: number) => (
            <details className="disclosure" key={i}>
              <summary>
                K = {r.k} · Loss {num(r.relative_loss ?? r.loss)}
              </summary>
              <Json value={r} />
            </details>
          ))}
        </>
      ) : (
        <Empty>此结果未保存 L3 数据</Empty>
      )}
      {value && Object.keys(value).length > 0 && (
        <details className="disclosure">
          <summary>部署身份与原始 L3 产物</summary>
          <Json value={value} />
        </details>
      )}
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
