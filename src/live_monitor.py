# src/live_monitor.py
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Group
from rich.text import Text
import numpy as np
import redis
import json
import threading
import logging
import time

logger = logging.getLogger(__name__)

class LiveMonitor:
    def __init__(self): # <-- [اصلاح] صف حذف شد
        self.layout = self._make_layout()
        self.agent_status = {}
        self.gen_num = 1
        self.total_gens = 0
        # self.queue = queue # <-- حذف شد
        self.live = Live(self.layout, screen=True, redirect_stderr=False, refresh_per_second=4)
        
        # --- [اصلاح کلیدی] اضافه کردن کلاینت Redis و PubSub ---
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe("trading_bot_monitor")
            self._stop_event = threading.Event()
            logger.info("LiveMonitor connected to Redis and subscribed to 'trading_bot_monitor'.")
        except redis.exceptions.ConnectionError as e:
            print(f"[ERROR] LiveMonitor: Could not connect to Redis. {e}")
            print("[ERROR] Please ensure Redis server is running on localhost:6379.")
            self.redis_client = None
            self.pubsub = None
            self._stop_event = threading.Event() # Still create event to avoid errors

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # --- [اصلاح کلیدی] سیگنال توقف و بستن اتصالات ---
        self._stop_event.set() # Signal the thread to stop
        self.live.stop()
        try:
            if self.pubsub:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            if self.redis_client:
                self.redis_client.close()
            logger.info("LiveMonitor connections to Redis closed.")
        except Exception:
            pass # Ignore errors on shutdown

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(size=11, name="footer")
        )
        layout["main"].split_row(Layout(name="side", size=45), Layout(name="body", ratio=2))
        layout["side"].split(Layout(name="progress"), Layout(name="gen_summary"))
        layout["footer"].split_row(Layout(name="top_agents"), Layout(name="params"))
        return layout

    def start_generation(self, pop_size: int, regime_type: str):
        # --- [اصلاح کلیدی] اضافه کردن شناسه رژیم به کلیدهای ایجنت ---
        regime_prefix = regime_type[:4].upper() # e.g., "TREN" or "RANG"
        
        self.agent_status = {
            f"g{self.gen_num}_{regime_prefix}_a{i}": { # <--- [اصلاح] استفاده از فرمت ID کامل
                "status": "Waiting", "step": "-", "eval_step": "-",
                "train_pnl": 0.0, "train_win_rate": 0.0,
            } for i in range(pop_size)
        }
        self._update_header()
        self._update_progress()
        self._update_agent_table()
    
    def _update_header(self):
        header_text = Text("Multi-Objective Trading Bot Training", justify="center", style="bold magenta")
        self.layout["header"].update(Panel(header_text))

    def _update_progress(self):
        gen_progress = Progress(TextColumn("[bold blue]Generations"), BarColumn(), TextColumn("{task.completed}/{task.total}"))
        gen_progress.add_task("Gen", total=self.total_gens, completed=self.gen_num - 1)
        self.layout["progress"].update(Panel(gen_progress, title="Overall Progress"))

    def _update_agent_table(self):
        table = Table(title=f"Agent Validation Status - Generation {self.gen_num}")
        table.add_column("ID", style="cyan", no_wrap=True, min_width=12)
        table.add_column("Status", min_width=12)
        table.add_column("Progress", style="yellow", justify="right")
        table.add_column("Calmar (Val)", style="bold magenta", justify="right")
        table.add_column("Drawdown (Val)", style="bold red", justify="right")
        table.add_column("PF (Val)", style="bold green", justify="right")
        table.add_column("Sharpe (Val)", style="bold yellow", justify="right")
        table.add_column("Win% (Val)", style="bold blue", justify="right")
        table.add_column("Trades", justify="right")

        status_map = {
            "Training": Text("Training", "yellow"), "Waiting": Text("Waiting", "dim"),
            "Initializing": Text("Initializing", "cyan"), "Evaluating": Text("Evaluating", "blue"),
            "Done": Text("Done", "bold green"), "Failed": Text("Failed", "bold red")
        }

        # --- [اصلاح جزئی] اضافه کردن شناسه رژیم به ID ایجنت‌ها برای خوانایی بهتر ---
        sorted_agents = sorted(self.agent_status.items(), 
                               key=lambda item: (int(item[0].split('_a')[1].split('_')[0]), item[0]))


        for agent_id, data in sorted_agents:
            status = data.get('status', 'N/A')
            final_metrics = data.get('final_metrics', {})
            is_done = (status == 'Done')

            progress_text = "-"
            if status == 'Training':
                progress_text = data.get('step', '-')
            elif status == 'Evaluating':
                progress_text = data.get('eval_step', '-')
            
            calmar_score = f"{final_metrics.get('calmar', 0):.2f}" if is_done else "-"
            drawdown = f"{final_metrics.get('drawdown', 0):.1f}" if is_done else "-"
            pf = f"{final_metrics.get('profit_factor', 0):.2f}" if is_done else "-"
            sharpe = f"{final_metrics.get('sharpe', 0):.2f}" if is_done else "-"
            win_rate = f"{final_metrics.get('win_rate', 0):.1f}" if is_done else "-"
            trades = f"{final_metrics.get('total_trades', 0)}" if is_done else "-"

            table.add_row(
                agent_id, status_map.get(status, Text(status)),
                progress_text, calmar_score, drawdown, pf, sharpe, win_rate, trades
            )
        self.layout["body"].update(Panel(table))

    def process_updates(self):
        """
        Processes updates from the Redis pub/sub channel.
        """
        if not self.pubsub:
             logger.error("Redis pubsub is not initialized. Monitor thread is stopping.")
             return

        while not self._stop_event.is_set():
            try:
                message = self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    continue
                
                update = json.loads(message['data'])

                if isinstance(update, dict) and update.get('type') == 'start_generation':
                    pop_size = update['pop_size']
                    regime_type = update.get('regime_type', 'N/A')
                    self.start_generation(pop_size, regime_type)
                    continue

                agent_id = update.get('id')
                if agent_id and agent_id in self.agent_status:
                    self.agent_status[agent_id].update(update)

                    # --- [اصلاح کلیدی] بررسی و ثبت خطا ---
                    if update.get('status') == 'Failed':
                        # لاگر مانیتور (که به فایل متصل است) خطا را ثبت می‌کند
                        error_details = update.get('traceback', update.get('error', 'Unknown error'))
                        logger.error(f"--- AGENT FAILED: {agent_id} ---\n{error_details}")

            except (redis.exceptions.ConnectionError, redis.exceptions.BusyLoadingError):
                logger.warning("Redis connection lost. Retrying in 5s...")
                time.sleep(5)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode monitor message: {message.get('data') if message else 'N/A'}")
            except Exception as e:
                logger.error(f"Error in monitor process_updates: {e}", exc_info=True)
                pass

            self._update_agent_table() 
        
        logger.info("Monitor process_updates thread received stop signal and is exiting.")


    def update_generation_summary(self, gen_num: int, results: list):
        # ... this method remains unchanged ...
        self.gen_num = gen_num + 1 
        self._update_progress()
        if not results: return
        valid_results = [r for r in results if r and r.get('total_trades', 0) > 0]
        if not valid_results: return
        pfs = np.array([r.get('profit_factor', 0) for r in valid_results])
        sharpes = np.array([r.get('sharpe', 0) for r in valid_results])
        dds = np.array([r.get('drawdown', 10000) for r in valid_results])
        summary_group = Group(
            Text(f"Avg Profit Factor: [yellow]{np.mean(pfs):.2f}[/] (Median: {np.median(pfs):.2f})"),
            Text(f"Avg Sharpe Ratio:  [yellow]{np.mean(sharpes):.2f}[/] (Median: {np.median(sharpes):.2f})"),
            Text(f"Avg Drawdown:      [yellow]{np.mean(dds):.1f}[/] pips (Median: {np.median(dds):.1f})"),
        )
        self.layout["gen_summary"].update(Panel(summary_group, title=f"Generation {gen_num} Validation Summary (Population Avg)"))
        sorted_results = sorted(valid_results, key=lambda r: -r.get('calmar', 0))
        params_table = Table(title=f"Top 3 Agents from Generation {gen_num} (by Validation Calmar)", title_style="bold blue", box=None)
        params_table.add_column("Calmar", style="magenta")
        params_table.add_column("Drawdown", style="red")
        params_table.add_column("PF", style="green")
        params_table.add_column("Trades", style="cyan")
        params_table.add_column("LR", style="green")
        params_table.add_column("Clip", style="yellow")
        params_table.add_column("SL", style="red")
        params_table.add_column("TP", style="green")
        params_table.add_column("N_Steps", style="blue")
        for res in sorted_results[:3]:
            if 'hyperparams' not in res or 'rl_params' not in res.get('hyperparams', {}):
                continue
            params = res['hyperparams']['rl_params']
            params_table.add_row(
                f"{res.get('calmar', 0):.2f}", f"{res.get('drawdown', 0):.1f}",
                f"{res.get('profit_factor', 0):.2f}", f"{res.get('total_trades', 0)}",
                f"{params.get('learning_rate', 0):.1e}", f"{params.get('clip_range', 0):.3f}",
                f"{params.get('stop_loss_atr_multiplier', 0):.2f}", f"{params.get('take_profit_atr_multiplier', 0):.2f}",
                f"{params.get('n_steps', 0)}"
            )
        self.layout["params"].update(Panel(params_table))