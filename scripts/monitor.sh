#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  monitor.sh — Live system monitor for exoplanet scanning
#
#  Run in a second tmux pane:   bash /opt/exoplanet/scripts/monitor.sh
#  Refreshes every 2 seconds.
#
#  Requires: lm-sensors (sensors), sysstat (iostat)
#  Install:  apt install -y lm-sensors sysstat
# ─────────────────────────────────────────────────────────────

INTERVAL=2
N_CORES=$(nproc)

# Colour codes
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_CYAN='\033[36m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_RED='\033[31m'
C_BLUE='\033[34m'
C_WHITE='\033[97m'

# Draw a bar [████░░░░] from a 0-100 value
bar() {
    local pct=${1:-0}
    local width=${2:-20}
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local bar=""
    local i
    for (( i=0; i<filled; i++ )); do bar+="█"; done
    for (( i=0; i<empty; i++ )); do bar+="░"; done
    printf "%s" "$bar"
}

# Colour a percentage value
pct_color() {
    local pct=${1:-0}
    if   (( pct >= 90 )); then printf '%s' "$C_RED"
    elif (( pct >= 70 )); then printf '%s' "$C_YELLOW"
    else                       printf '%s' "$C_GREEN"
    fi
}

# Temperature colour
temp_color() {
    local t=${1:-0}
    if   (( t >= 85 )); then printf '%s' "$C_RED"
    elif (( t >= 70 )); then printf '%s' "$C_YELLOW"
    else                     printf '%s' "$C_GREEN"
    fi
}

# Previous net/disk stats (for delta calculations)
PREV_NET_RX=0; PREV_NET_TX=0
PREV_DISK_R=0; PREV_DISK_W=0
PREV_TS=0

get_net_stats() {
    # Sum rx bytes (field 2) and tx bytes (field 10) for all non-loopback interfaces
    # /proc/net/dev: "  iface: rxbytes rxpkts ... txbytes txpkts ..."
    awk '/:/ && !/lo:/ { gsub(/:/, " "); rx += $2; tx += $10 } END { printf "%d %d\n", rx+0, tx+0 }' /proc/net/dev
}

get_disk_stats() {
    # Sum sectors read (field 6) and written (field 10) from real block devices
    # /proc/diskstats fields: major minor name rc rm rs rt wc wm ws wt iop tot wtot ...
    awk 'NF>=14 && $3 !~ /^(loop|dm|sr)/ { r += $6; w += $10 } END { printf "%d %d\n", r+0, w+0 }' /proc/diskstats
}

get_temperatures_sensors() {
    # Use lm-sensors if available, fall back to hwmon
    if command -v sensors &>/dev/null; then
        # Parse 'sensors' output for temperature lines
        # Lines look like: "Core 0:        +42.0°C  (high = +86.0°C, crit = +100.0°C)"
        # or:              "Package id 0:  +44.0°C  (high = +86.0°C, crit = +100.0°C)"
        local count=0
        while IFS= read -r line; do
            # Match lines with temperature values like +42.0°C or +42.0 C
            if [[ "$line" =~ ([A-Za-z][^:]+):[[:space:]]+\+([0-9]+\.[0-9]+)[[:space:]]*.?C ]]; then
                local label="${BASH_REMATCH[1]// /}"
                local temp_val="${BASH_REMATCH[2]}"
                local temp_int="${temp_val%%.*}"
                local color
                color=$(temp_color "$temp_int")
                printf "  %-22s ${color}%s°C${C_RESET}\n" "${label}:" "${temp_val}"
                (( count++ ))
                (( count >= 10 )) && break
            fi
        done < <(sensors 2>/dev/null)
        if (( count == 0 )); then
            get_temperatures_hwmon
        fi
    else
        get_temperatures_hwmon
    fi
}

get_temperatures_hwmon() {
    # Fallback: read directly from /sys/class/hwmon
    local found=0
    for f in /sys/class/hwmon/hwmon*/temp*_input; do
        [[ -f "$f" ]] || continue
        local label_f="${f/_input/_label}"
        local label="temp"
        [[ -f "$label_f" ]] && label=$(cat "$label_f")
        local val=$(( $(cat "$f") / 1000 ))
        local color
        color=$(temp_color "$val")
        printf "  %-22s ${color}%d°C${C_RESET}\n" "${label}:" "$val"
        (( found++ ))
        (( found >= 6 )) && break
    done
    if (( found == 0 )); then
        printf "  [temperatures not available — install lm-sensors]\n"
    fi
}

get_disk_util_iostat() {
    # Use iostat -x 1 1 if available (sysstat package)
    if command -v iostat &>/dev/null; then
        # iostat -x gives extended stats including %util
        # Parse the device lines (skip loop, dm, sr devices)
        local count=0
        while IFS= read -r line; do
            # Match device lines: start with a device name followed by numbers
            if [[ "$line" =~ ^([a-z]+[a-z0-9]+)[[:space:]]+[0-9] ]]; then
                local dev="${BASH_REMATCH[1]}"
                [[ "$dev" =~ ^(loop|dm|sr) ]] && continue
                # Extract %util (last field in extended iostat output)
                local util
                util=$(echo "$line" | awk '{print $NF}')
                # Validate it's a number
                if [[ "$util" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                    local util_int="${util%%.*}"
                    local color
                    color=$(pct_color "$util_int")
                    printf "  %-10s util: ${color}%s${C_RESET}  %5.1f%%\n" \
                           "$dev" "$(bar "$util_int" 16)" "$util"
                    (( count++ ))
                fi
                (( count >= 4 )) && break
            fi
        done < <(iostat -x 1 1 2>/dev/null | grep -v '^$' | grep -v Device | grep -v Linux | grep -v avg-cpu)
        if (( count == 0 )); then
            printf "  [disk util not available — install sysstat]\n"
        fi
    else
        printf "  [iostat not found — install sysstat]\n"
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────
while true; do
    NOW=$(date +%s%N)
    DELTA_S=$(( (NOW - PREV_TS) / 1000000000 ))
    [[ $DELTA_S -eq 0 ]] && DELTA_S=1

    # ── CPU per-core usage via /proc/stat ──────────────────────────────────
    declare -a CPU_PCT
    # Note: /proc/stat has 10+ fields (steal guest guest_nice ...) so parse with awk
    while read -r cpu_name user nice sys idle iowait irq softirq steal _rest; do
        [[ "$cpu_name" =~ ^cpu[0-9]+$ ]] || continue
        local_n="${cpu_name#cpu}"
        steal=${steal:-0}
        total=$(( user + nice + sys + idle + iowait + irq + softirq + steal ))
        busy=$(( total - idle - iowait ))
        prev_total_var="PREV_T${local_n}"
        prev_busy_var="PREV_B${local_n}"
        prev_t="${!prev_total_var:-0}"
        prev_b="${!prev_busy_var:-0}"
        dt=$(( total - prev_t ))
        db=$(( busy - prev_b ))
        if (( dt > 0 )); then
            pct=$(( 100 * db / dt ))
        else
            pct=0
        fi
        CPU_PCT[$local_n]=$pct
        declare "PREV_T${local_n}=$total"
        declare "PREV_B${local_n}=$busy"
    done < /proc/stat

    # ── RAM ────────────────────────────────────────────────────────────────
    read -r MEM_TOTAL MEM_FREE MEM_AVAIL < <(
        awk '/MemTotal/{t=$2} /MemFree/{f=$2} /MemAvailable/{a=$2}
             END{printf "%d %d %d", t, f, a}' /proc/meminfo
    )
    MEM_USED=$(( MEM_TOTAL - MEM_AVAIL ))
    MEM_PCT=$(( 100 * MEM_USED / MEM_TOTAL ))
    MEM_USED_GB=$(awk "BEGIN{printf \"%.1f\", $MEM_USED/1048576}")
    MEM_TOTAL_GB=$(awk "BEGIN{printf \"%.1f\", $MEM_TOTAL/1048576}")

    # ── Network I/O ────────────────────────────────────────────────────────
    read -r CUR_RX CUR_TX < <(get_net_stats)
    NET_RX_KB=$(( (CUR_RX - PREV_NET_RX) / DELTA_S / 1024 ))
    NET_TX_KB=$(( (CUR_TX - PREV_NET_TX) / DELTA_S / 1024 ))
    PREV_NET_RX=$CUR_RX; PREV_NET_TX=$CUR_TX

    # ── Disk I/O ───────────────────────────────────────────────────────────
    read -r CUR_DR CUR_DW < <(get_disk_stats)
    DISK_R_MB=$(awk "BEGIN{printf \"%.1f\", ($CUR_DR - $PREV_DISK_R) * 512 / $DELTA_S / 1048576}")
    DISK_W_MB=$(awk "BEGIN{printf \"%.1f\", ($CUR_DW - $PREV_DISK_W) * 512 / $DELTA_S / 1048576}")
    PREV_DISK_R=$CUR_DR; PREV_DISK_W=$CUR_DW
    PREV_TS=$NOW

    # ── Draw screen ────────────────────────────────────────────────────────
    clear
    echo -e "${C_BOLD}${C_CYAN}══════════ System Monitor — $(date '+%H:%M:%S') ══════════${C_RESET}"
    echo

    # CPU cores
    echo -e "${C_BOLD}${C_WHITE}  CPU Cores  (${N_CORES} cores)${C_RESET}"
    for (( i=0; i<N_CORES; i++ )); do
        pct=${CPU_PCT[$i]:-0}
        color=$(pct_color $pct)
        printf "  Core %2d  ${color}%s${C_RESET}  %3d%%\n" \
               "$i" "$(bar $pct 24)" "$pct"
    done
    echo

    # RAM
    echo -e "${C_BOLD}${C_WHITE}  Memory${C_RESET}"
    ram_color=$(pct_color $MEM_PCT)
    printf "  RAM      ${ram_color}%s${C_RESET}  %3d%%  (%s / %s GB)\n" \
           "$(bar $MEM_PCT 24)" "$MEM_PCT" "$MEM_USED_GB" "$MEM_TOTAL_GB"
    echo

    # Temperatures (via sensors)
    echo -e "${C_BOLD}${C_WHITE}  Temperatures${C_RESET}"
    get_temperatures_sensors
    echo

    # Disk utilization (via iostat)
    echo -e "${C_BOLD}${C_WHITE}  Disk Utilization${C_RESET}"
    get_disk_util_iostat
    echo

    # I/O throughput
    echo -e "${C_BOLD}${C_WHITE}  I/O Throughput${C_RESET}"
    printf "  Disk     read: %6.1f MB/s   write: %6.1f MB/s\n" \
           "${DISK_R_MB:-0}" "${DISK_W_MB:-0}"
    printf "  Network  ↓ %6d KB/s      ↑ %6d KB/s\n" \
           "$NET_RX_KB" "$NET_TX_KB"
    echo

    echo -e "${C_DIM}  Refreshes every ${INTERVAL}s — Ctrl-C to exit${C_RESET}"

    sleep "$INTERVAL"
done
