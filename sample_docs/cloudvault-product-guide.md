# CloudVault Backup — Product Guide

*Fictional sample document for demonstration purposes.*

## Overview

CloudVault is Northwind Labs' managed backup service for small and medium businesses. It backs up laptops, servers and Microsoft 365 or Google Workspace accounts to encrypted cloud storage, and lets administrators restore single files or whole machines from a web console.

## Plans and pricing

| Plan | Price | Storage | Devices | Retention |
|---|---|---|---|---|
| Starter | USD 8 per device per month | 250 GB per device | Up to 10 | 30 days |
| Business | USD 15 per device per month | 1 TB per device | Up to 250 | 1 year |
| Enterprise | Custom pricing | Unlimited | Unlimited | Up to 7 years |

Annual billing gives a 15% discount on Starter and Business plans. All plans include a 14-day free trial with no credit card required.

## Security

All backups are encrypted with AES-256 before they leave the device, and data in transit uses TLS 1.3. Business and Enterprise customers can choose a private encryption key, which means not even Northwind Labs can read their data; if a customer loses the private key, the data cannot be recovered. CloudVault is certified to ISO 27001 and completes a SOC 2 Type II audit every year.

Data is stored in the customer's chosen region: EU (Frankfurt), US (Virginia) or Middle East (UAE). Data never leaves the chosen region unless the customer moves it.

## Backup schedule

By default, devices are backed up every 4 hours while they are online. Administrators can change the schedule to anything between every hour and once per day. Backups are incremental: after the first full backup, only changed blocks are uploaded, which keeps bandwidth use low.

## Restores and service levels

Single files can be restored instantly from the web console. A full machine restore downloads a recovery image; on a 100 Mbps connection, restoring 100 GB takes about 2.5 hours. Enterprise customers can request a physical drive with their data shipped within 2 business days.

Service availability targets are 99.9% for Starter and Business and 99.95% for Enterprise. If monthly availability falls below the target, customers receive a service credit of 10% of that month's fee for each 0.1% below target, up to 50%.

## Support

Starter customers get email support with a response within one business day. Business customers get email and chat support with a response within 4 hours, 24/7. Enterprise customers also get a named technical account manager and a 1-hour response time for critical incidents.

## Cancellation

Customers can cancel at any time from the billing page. After cancellation, backups remain available for download for 30 days and are then permanently deleted. Annual plans are refunded pro rata for unused full months.
