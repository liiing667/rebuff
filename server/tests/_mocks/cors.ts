// cors middleware fake: just invoke the handler.
export default function cors() {
  return (_req: unknown, _res: unknown, next: () => void) => next();
}
