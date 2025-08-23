# Runbook: PIR Server Failure

## Alert Name
`PIRServerDown`

## Alert Description
One or more PIR servers are not responding or marked as unhealthy.

## Severity
**CRITICAL** - Privacy guarantees may be compromised if multiple servers fail.

## Impact
- With 1 server down: System continues with degraded redundancy
- With 2+ servers down: **PRIVACY GUARANTEES VIOLATED** - System cannot maintain information-theoretic privacy

## Dashboard Links
- [PIR Server Performance](https://grafana.genomevault.io/d/genomevault-pir)
- [Privacy Metrics](https://grafana.genomevault.io/d/genomevault-privacy)

## Verification Steps

1. **Check server health status:**
```bash
kubectl get pods -n pir-server-1
kubectl get pods -n pir-server-2
kubectl get pods -n pir-server-3
```

2. **Check recent logs:**
```bash
kubectl logs -n pir-server-1 -l app=pir-server --tail=100
kubectl logs -n pir-server-2 -l app=pir-server --tail=100
```

3. **Verify network connectivity:**
```bash
kubectl exec -n pir-coordinator deployment/pir-coordinator -- \
  curl -s http://pir-server-1.pir-server-1:8081/health/ready
```

4. **Check metrics:**
```bash
curl -s http://prometheus:9090/api/v1/query?query=genomevault_pir_server_health
```

## Immediate Actions

### If 1 Server is Down:

1. **Identify the failed server:**
```bash
kubectl get pods -n pir-server-<ID> -o wide
```

2. **Check for OOM kills or restarts:**
```bash
kubectl describe pod -n pir-server-<ID> <pod-name>
```

3. **Force restart if necessary:**
```bash
kubectl rollout restart deployment/pir-server-<ID> -n pir-server-<ID>
```

### If 2+ Servers are Down (CRITICAL):

1. **IMMEDIATELY notify security team** - Privacy guarantees are violated

2. **Check for coordinated attack:**
```bash
# Check for unusual network activity
kubectl top pods -n pir-server-1
kubectl top pods -n pir-server-2
kubectl top pods -n pir-server-3
```

3. **Enable emergency mode to prevent data access:**
```bash
kubectl scale deployment/pir-coordinator --replicas=0 -n pir-coordinator
```

4. **Investigate root cause:**
   - Check for configuration changes
   - Review recent deployments
   - Check infrastructure issues

## Recovery Steps

### Single Server Recovery:

1. **Check pod events:**
```bash
kubectl get events -n pir-server-<ID> --sort-by='.lastTimestamp'
```

2. **If persistent failures, check PVC:**
```bash
kubectl get pvc -n pir-server-<ID>
kubectl describe pvc pir-server-<ID>-data -n pir-server-<ID>
```

3. **Verify configuration:**
```bash
kubectl get configmap -n pir-server-<ID> pir-server-config -o yaml
```

4. **Scale up replicas if needed:**
```bash
kubectl scale deployment/pir-server-<ID> --replicas=2 -n pir-server-<ID>
```

### Multiple Server Recovery:

1. **Check cluster-wide issues:**
```bash
kubectl get nodes
kubectl top nodes
```

2. **Verify network policies:**
```bash
kubectl get networkpolicy -n pir-server-1
kubectl get networkpolicy -n pir-server-2
```

3. **Restore from backup if corruption detected:**
```bash
# Stop servers
kubectl scale deployment/pir-server-1 --replicas=0 -n pir-server-1
kubectl scale deployment/pir-server-2 --replicas=0 -n pir-server-2

# Restore data
./scripts/restore_pir_data.sh --server-id=1 --backup-id=latest
./scripts/restore_pir_data.sh --server-id=2 --backup-id=latest

# Restart servers
kubectl scale deployment/pir-server-1 --replicas=2 -n pir-server-1
kubectl scale deployment/pir-server-2 --replicas=2 -n pir-server-2
```

## Root Cause Analysis

Common causes:
1. **Memory exhaustion** - Check resource limits
2. **Disk full** - Check PVC usage
3. **Network partition** - Check network policies
4. **Byzantine behavior** - Check for malicious activity
5. **Configuration drift** - Compare configs across servers

## Prevention

1. **Implement resource monitoring:**
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

2. **Set up PVC monitoring:**
```bash
kubectl patch pvc pir-server-<ID>-data -n pir-server-<ID> \
  -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

3. **Enable automatic scaling:**
```bash
kubectl autoscale deployment pir-server-<ID> \
  --min=2 --max=4 --cpu-percent=70 -n pir-server-<ID>
```

## Escalation

If the issue persists after 30 minutes:
1. Page on-call SRE team
2. Notify security team if 2+ servers affected
3. Prepare incident report for compliance

## Related Runbooks
- [Byzantine Fault Detection](./byzantine-fault.md)
- [PIR High Latency](./pir-high-latency.md)
- [Privacy Guarantee Risk](./privacy-guarantee-risk.md)

## Compliance Notes

**HIPAA Requirements:**
- All troubleshooting actions must be logged
- PHI access during debugging must be audited
- Incident must be documented within 24 hours
- Risk assessment required if privacy breach suspected

## Contacts

- **On-Call SRE:** PagerDuty group `genomevault-sre`
- **Security Team:** `security@genomevault.io`
- **Compliance Officer:** `compliance@genomevault.io`

## Change Log

| Date | Author | Description |
|------|--------|-------------|
| 2024-01-15 | SRE Team | Initial runbook creation |
| 2024-01-20 | Security | Added privacy breach procedures |
