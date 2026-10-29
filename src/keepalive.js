export default {
  async scheduled(event, env, ctx) {
    const url = env.STREAMLIT_APP_URL;
    ctx.waitUntil(
      fetch(url, { cf: { cacheEverything: false } }).then((res) => {
        console.log(`Keep-alive ping: ${res.status}`);
      }),
    );
  },
};
