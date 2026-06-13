import { jetstream, jetstreamManager } from "@nats-io/jetstream";
import { connect, deferred, nuid } from "@nats-io/transport-node";

const stream = 'one';
const subj = 'one';
//const consumer = 'consumer1'

const nc = await connect({ servers: "nats://127.0.0.1:4222" });
console.log(`connected`);

const js = jetstream(nc);
const jsm = await jetstreamManager(nc);

//nc.publish(subj, 'hellojs')
let pa = await js.publish(subj, "hello")
const streamname = pa.stream
const seq = pa.seq
const duplicate = pa.duplicate
console.log(streamname, seq, duplicate)

/*
const sm = await jsm.streams.getMessage(stream, { seq: 2 });
console.log(sm.seq);

await jsm.streams.deleteMessage(stream, 5);
*/


//const c = await js.consumers.get(stream, consumer)

const c = await js.consumers.get(stream)

/*
const m = await c.next();
if (m) {
  console.log(m.subject);
  m.ack();
} else {
  console.log(`didn't get a message`);
}
*/


//console.log(c)

const messages = await c.consume({ max_messages: 1 });
for await (const m of messages) {
  console.log(m.seq, m.string());
  m.ack();
  await jsm.streams.deleteMessage(stream, m.seq);
}

await nc.drain();
