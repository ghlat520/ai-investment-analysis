/**
 * 根据股票代码和市场生成东方财富行情页链接
 */
export function getEastMoneyUrl(symbol: string, market?: string): string {
  if (market === 'HK' || /^\d{5}$/.test(symbol)) {
    return `https://quote.eastmoney.com/hk/${symbol}.html`;
  }
  if (market === 'US' || /^[A-Za-z]+$/.test(symbol)) {
    return `https://quote.eastmoney.com/us/${symbol}.html`;
  }
  // A股：6/9开头=沪市，其余=深市
  const prefix = symbol.startsWith('6') || symbol.startsWith('9') ? 'SH' : 'SZ';
  return `https://quote.eastmoney.com/${prefix}${symbol}.html`;
}
